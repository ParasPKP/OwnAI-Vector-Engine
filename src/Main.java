import com.sun.net.httpserver.*;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;
import java.util.function.*;

/**
 * VectorDB — Vector Database from Scratch in Java
 * Implements HNSW, KD-Tree, and Brute Force search algorithms.
 * Includes a RAG pipeline powered by Ollama (local LLM).
 *
 * Compile: javac Main.java
 * Run:     java Main
 * Open:    http://localhost:8080
 */
public class Main {

    // ─── CONSTANTS ────────────────────────────────────────────────────────────
    static final int PORT = 8080;
    static final int DIMS = 16; // demo vectors are 16-dimensional

    // =========================================================================
    //  DISTANCE METRICS
    // =========================================================================

    /** Euclidean distance: straight-line distance between two points */
    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return (float) Math.sqrt(s);
    }

    /** Cosine distance: measures angle between vectors (0=same, 1=opposite) */
    static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
    }

    /** Manhattan distance: sum of absolute differences (like city blocks) */
    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    @FunctionalInterface
    interface DistFn { float dist(float[] a, float[] b); }

    static DistFn getDistFn(String m) {
        if ("cosine".equals(m))    return Main::cosine;
        if ("manhattan".equals(m)) return Main::manhattan;
        return Main::euclidean;
    }

    // =========================================================================
    //  VECTOR ITEM  — one row in the database
    // =========================================================================

    static class VectorItem {
        int     id;
        String  metadata, category;
        float[] emb;

        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id; this.metadata = metadata;
            this.category = category; this.emb = emb;
        }
    }

    // =========================================================================
    //  BRUTE FORCE  — O(N·d) exact search, baseline algorithm
    // =========================================================================

    static class BruteForce {
        final List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem v) { items.add(v); }

        /** Returns list of float[2] = {distance, id}, sorted nearest-first */
        List<float[]> knn(float[] q, int k, DistFn dist) {
            List<float[]> r = new ArrayList<>();
            for (VectorItem v : items)
                r.add(new float[]{ dist.dist(q, v.emb), v.id });
            r.sort(Comparator.comparingDouble(x -> x[0]));
            return r.subList(0, Math.min(k, r.size()));
        }

        void remove(int id) { items.removeIf(v -> v.id == id); }
    }

    // =========================================================================
    //  KD-TREE  — O(log N) exact search via axis-aligned space partitioning
    // =========================================================================

    static class KDTree {

        static class KDNode {
            VectorItem item;
            KDNode left, right;
            KDNode(VectorItem v) { item = v; }
        }

        KDNode root;
        final int dims;

        KDTree(int dims) { this.dims = dims; }

        private KDNode ins(KDNode n, VectorItem v, int depth) {
            if (n == null) return new KDNode(v);
            int ax = depth % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left  = ins(n.left,  v, depth + 1);
            else                              n.right = ins(n.right, v, depth + 1);
            return n;
        }

        void insert(VectorItem v) { root = ins(root, v, 0); }

        // Max-heap: we keep k closest, heap top = worst (largest) distance
        private void knnSearch(KDNode n, float[] q, int k, int depth, DistFn dist,
                               PriorityQueue<float[]> heap) {
            if (n == null) return;
            float dn = dist.dist(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek()[0]) {
                heap.offer(new float[]{ dn, n.item.id });
                if (heap.size() > k) heap.poll();
            }
            int ax = depth % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer  = diff < 0 ? n.left  : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knnSearch(closer,  q, k, depth + 1, dist, heap);
            // Only explore the far side if it could contain a closer point
            if (heap.size() < k || Math.abs(diff) < heap.peek()[0])
                knnSearch(farther, q, k, depth + 1, dist, heap);
        }

        List<float[]> knn(float[] q, int k, DistFn dist) {
            // Max-heap ordered by distance (largest on top, so we can prune)
            PriorityQueue<float[]> heap =
                    new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));
            knnSearch(root, q, k, 0, dist, heap);
            List<float[]> r = new ArrayList<>(heap);
            r.sort(Comparator.comparingDouble(x -> x[0]));
            return r;
        }

        void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
    }

    // =========================================================================
    //  HNSW — Hierarchical Navigable Small World  (production-grade ANN index)
    //
    //  Same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.
    //  Multilayer graph: upper layers = sparse "highway", layer 0 = dense.
    //  Insert/Search both run in O(log N).
    // =========================================================================

    static class HNSW {

        static class Node {
            VectorItem       item;
            int              maxLyr;
            List<List<Integer>> nbrs; // nbrs.get(layer) = neighbor IDs at that layer
        }

        final Map<Integer, Node> G = new HashMap<>();
        final int   M;       // max neighbors per layer (default 16)
        final int   M0;      // max neighbors at layer 0 = 2*M
        final int   efBuild; // beam width during construction (default 200)
        final float mL;      // level generation factor = 1/ln(M)
        int topLayer = -1;
        int entryPt  = -1;
        final Random rng = new Random(42);

        HNSW(int m, int efBuild) {
            M = m; M0 = 2 * m; this.efBuild = efBuild;
            mL = 1.0f / (float) Math.log(m);
        }

        /** Randomly assign a layer to a new node (exponential distribution) */
        private int randLevel() {
            return (int) Math.floor(-Math.log(Math.max(rng.nextFloat(), 1e-9f)) * mL);
        }

        /**
         * Greedy beam search within one layer.
         * Returns list of {distance, id} sorted nearest-first, up to ef results.
         */
        private List<float[]> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> visited = new HashSet<>();
            // candidates: min-heap (smallest distance first)
            PriorityQueue<float[]> cands =
                    new PriorityQueue<>(Comparator.comparingDouble(x -> x[0]));
            // found: max-heap (largest distance on top, for easy pruning)
            PriorityQueue<float[]> found =
                    new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

            float d0 = dist.dist(q, G.get(ep).item.emb);
            visited.add(ep);
            cands.offer(new float[]{ d0, ep });
            found.offer(new float[]{ d0, ep });

            while (!cands.isEmpty()) {
                float[] top = cands.poll();
                float cd = top[0]; int cid = (int) top[1];
                if (found.size() >= ef && cd > found.peek()[0]) break;
                Node cn = G.get(cid);
                if (cn == null || lyr >= cn.nbrs.size()) continue;
                for (int nid : cn.nbrs.get(lyr)) {
                    if (visited.contains(nid) || !G.containsKey(nid)) continue;
                    visited.add(nid);
                    float nd = dist.dist(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek()[0]) {
                        cands.offer(new float[]{ nd, nid });
                        found.offer(new float[]{ nd, nid });
                        if (found.size() > ef) found.poll();
                    }
                }
            }

            List<float[]> res = new ArrayList<>(found);
            res.sort(Comparator.comparingDouble(x -> x[0]));
            return res;
        }

        private List<Integer> selectNbrs(List<float[]> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++)
                r.add((int) cands.get(i)[1]);
            return r;
        }

        void insert(VectorItem item, DistFn dist) {
            int id  = item.id;
            int lvl = randLevel();

            Node node = new Node();
            node.item   = item;
            node.maxLyr = lvl;
            node.nbrs   = new ArrayList<>();
            for (int i = 0; i <= lvl; i++) node.nbrs.add(new ArrayList<>());
            G.put(id, node);

            if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

            // Descend from top layer to lvl+1 (coarse navigation)
            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                Node epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<float[]> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = (int) W.get(0)[1];
                }
            }

            // Descend from min(topLayer,lvl) to 0, connecting neighbors
            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<float[]>  W    = searchLayer(item.emb, ep, efBuild, lc, dist);
                int            maxM = (lc == 0) ? M0 : M;
                List<Integer>  sel  = selectNbrs(W, maxM);
                node.nbrs.get(lc).addAll(sel);

                // Bidirectional connection + prune to maxM
                for (int nid : sel) {
                    if (!G.containsKey(nid)) continue;
                    Node nn = G.get(nid);
                    while (nn.nbrs.size() <= lc) nn.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nn.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<float[]> ds = new ArrayList<>();
                        for (int c : conn)
                            if (G.containsKey(c))
                                ds.add(new float[]{ dist.dist(nn.item.emb, G.get(c).item.emb), c });
                        ds.sort(Comparator.comparingDouble(x -> x[0]));
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++)
                            conn.add((int) ds.get(i)[1]);
                    }
                }
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
            if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
        }

        List<float[]> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                Node epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<float[]> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = (int) W.get(0)[1];
                }
            }
            List<float[]> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            return new ArrayList<>(W.subList(0, Math.min(k, W.size())));
        }

        void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values())
                for (List<Integer> layer : nd.nbrs)
                    layer.removeIf(nid -> nid == id);
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet()) if (nid != id) { entryPt = nid; break; }
            }
            G.remove(id);
        }

        /** Returns graph metadata for the /hnsw-info API endpoint */
        Map<String, Object> getInfo() {
            int maxL = Math.max(topLayer + 1, 1);
            int[] nodesPerLayer = new int[maxL];
            int[] edgesPerLayer = new int[maxL];
            List<Map<String, Object>> nodes = new ArrayList<>();
            List<Map<String, Object>> edges = new ArrayList<>();

            for (Map.Entry<Integer, Node> entry : G.entrySet()) {
                int id = entry.getKey(); Node nd = entry.getValue();
                Map<String, Object> nm = new LinkedHashMap<>();
                nm.put("id", id); nm.put("metadata", nd.item.metadata);
                nm.put("category", nd.item.category); nm.put("maxLyr", nd.maxLyr);
                nodes.add(nm);
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    nodesPerLayer[lc]++;
                    if (lc < nd.nbrs.size())
                        for (int nid : nd.nbrs.get(lc))
                            if (id < nid) {
                                edgesPerLayer[lc]++;
                                Map<String, Object> em = new LinkedHashMap<>();
                                em.put("src", id); em.put("dst", nid); em.put("lyr", lc);
                                edges.add(em);
                            }
                }
            }
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("topLayer", topLayer); info.put("nodeCount", G.size());
            info.put("nodesPerLayer", nodesPerLayer); info.put("edgesPerLayer", edgesPerLayer);
            info.put("nodes", nodes); info.put("edges", edges);
            return info;
        }

        int size() { return G.size(); }
    }

    // =========================================================================
    //  VECTOR DB  — unified 16D demo index (BruteForce + KDTree + HNSW)
    // =========================================================================

    static class VectorDB {
        final Map<Integer, VectorItem> store = new HashMap<>();
        final BruteForce  bf   = new BruteForce();
        final KDTree      kdt;
        final HNSW        hnsw = new HNSW(16, 200);
        final ReentrantLock mu = new ReentrantLock();
        int nextId = 1;
        final int dims;

        VectorDB(int d) { dims = d; kdt = new KDTree(d); }

        int insert(String meta, String cat, float[] emb, DistFn dist) {
            mu.lock();
            try {
                VectorItem v = new VectorItem(nextId++, meta, cat, emb);
                store.put(v.id, v);
                bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
                return v.id;
            } finally { mu.unlock(); }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id); bf.remove(id); hnsw.remove(id);
                kdt.rebuild(new ArrayList<>(store.values()));
                return true;
            } finally { mu.unlock(); }
        }

        static class Hit {
            int id; String meta, cat; float[] emb; float dist;
            Hit(int id, String meta, String cat, float[] emb, float dist) {
                this.id=id; this.meta=meta; this.cat=cat; this.emb=emb; this.dist=dist;
            }
        }
        static class SearchOut { List<Hit> hits; long us; String algo, metric; }

        SearchOut search(float[] q, int k, String metric, String algo) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                long t0 = System.nanoTime();
                List<float[]> raw;
                if      ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
                else if ("kdtree".equals(algo))     raw = kdt.knn(q, k, dfn);
                else                                raw = hnsw.knn(q, k, 50, dfn);
                long us = (System.nanoTime() - t0) / 1000;
                SearchOut out = new SearchOut();
                out.hits = new ArrayList<>(); out.us = us; out.algo = algo; out.metric = metric;
                for (float[] r : raw) {
                    int id = (int) r[1];
                    if (store.containsKey(id)) {
                        VectorItem v = store.get(id);
                        out.hits.add(new Hit(id, v.metadata, v.category, v.emb, r[0]));
                    }
                }
                return out;
            } finally { mu.unlock(); }
        }

        long[] benchmark(float[] q, int k, String metric) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                long t1 = System.nanoTime(); bf.knn(q, k, dfn);       long bfUs   = (System.nanoTime() - t1) / 1000;
                long t2 = System.nanoTime(); kdt.knn(q, k, dfn);      long kdUs   = (System.nanoTime() - t2) / 1000;
                long t3 = System.nanoTime(); hnsw.knn(q, k, 50, dfn); long hnswUs = (System.nanoTime() - t3) / 1000;
                return new long[]{ bfUs, kdUs, hnswUs, store.size() };
            } finally { mu.unlock(); }
        }

        List<VectorItem> all() {
            mu.lock(); try { return new ArrayList<>(store.values()); } finally { mu.unlock(); }
        }

        Map<String, Object> hnswInfo() {
            mu.lock(); try { return hnsw.getInfo(); } finally { mu.unlock(); }
        }

        int size() { mu.lock(); try { return store.size(); } finally { mu.unlock(); } }
    }

    // =========================================================================
    //  DOCUMENT ITEM  — one chunk stored in DocumentDB
    // =========================================================================

    static class DocItem {
        int id; String title, text; float[] emb;
        DocItem(int id, String title, String text, float[] emb) {
            this.id=id; this.title=title; this.text=text; this.emb=emb;
        }
    }

    // =========================================================================
    //  DOCUMENT DB  — HNSW index over real 768D Ollama embeddings
    // =========================================================================

    static class DocumentDB {
        final Map<Integer, DocItem> store = new HashMap<>();
        final HNSW        hnsw = new HNSW(16, 200);
        final BruteForce  bf   = new BruteForce();
        final ReentrantLock mu = new ReentrantLock();
        int nextId = 1;
        int dims   = 0;

        int insert(String title, String text, float[] emb) {
            mu.lock();
            try {
                if (dims == 0) dims = emb.length;
                DocItem item = new DocItem(nextId++, title, text, emb);
                store.put(item.id, item);
                VectorItem vi = new VectorItem(item.id, title, "doc", emb);
                hnsw.insert(vi, Main::cosine);
                bf.insert(vi);
                return item.id;
            } finally { mu.unlock(); }
        }

        List<float[]> search(float[] q, int k) {
            mu.lock();
            try {
                if (store.isEmpty()) return new ArrayList<>();
                List<float[]> raw = (store.size() < 10)
                        ? bf.knn(q, k, Main::cosine)
                        : hnsw.knn(q, k, 50, Main::cosine);
                List<float[]> out = new ArrayList<>();
                for (float[] r : raw)
                    if (r[0] <= 0.7f && store.containsKey((int) r[1]))
                        out.add(r);
                return out;
            } finally { mu.unlock(); }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id); hnsw.remove(id); bf.remove(id);
                return true;
            } finally { mu.unlock(); }
        }

        List<DocItem> all() {
            mu.lock(); try { return new ArrayList<>(store.values()); } finally { mu.unlock(); }
        }

        DocItem get(int id) {
            mu.lock(); try { return store.get(id); } finally { mu.unlock(); }
        }

        int size()    { mu.lock(); try { return store.size(); } finally { mu.unlock(); } }
        int getDims() { return dims; }
    }

    // =========================================================================
    //  OLLAMA CLIENT  — talks to the local Ollama REST API
    //  Install Ollama: https://ollama.com
    //  Pull models:    ollama pull nomic-embed-text
    //                  ollama pull llama3.2
    // =========================================================================

    static class OllamaClient {
        String embedModel = "nomic-embed-text";
        String genModel   = "llama3.2";
        final String baseUrl = "http://127.0.0.1:11434";

        final HttpClient http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3)).build();

        boolean isAvailable() {
            try {
                HttpResponse<String> r = http.send(
                        HttpRequest.newBuilder().uri(URI.create(baseUrl + "/api/tags"))
                                .timeout(Duration.ofSeconds(2)).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
                return r.statusCode() == 200;
            } catch (Exception e) { return false; }
        }

        float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + escJson(text) + "\"}";
                HttpResponse<String> r = http.send(
                        HttpRequest.newBuilder().uri(URI.create(baseUrl + "/api/embeddings"))
                                .timeout(Duration.ofSeconds(60))
                                .header("Content-Type", "application/json")
                                .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                        HttpResponse.BodyHandlers.ofString());
                if (r.statusCode() != 200) return new float[0];
                return parseEmbedding(r.body());
            } catch (Exception e) { return new float[0]; }
        }

        String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\","
                        + "\"prompt\":\"" + escJson(prompt) + "\","
                        + "\"stream\":false}";
                HttpResponse<String> r = http.send(
                        HttpRequest.newBuilder().uri(URI.create(baseUrl + "/api/generate"))
                                .timeout(Duration.ofSeconds(180))
                                .header("Content-Type", "application/json")
                                .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                        HttpResponse.BodyHandlers.ofString());
                if (r.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
                return extractStr(r.body(), "response");
            } catch (Exception e) { return "ERROR: " + e.getMessage(); }
        }

        private float[] parseEmbedding(String body) {
            int p = body.indexOf("\"embedding\"");
            if (p < 0) return new float[0];
            p = body.indexOf('[', p);
            if (p < 0) return new float[0];
            int depth = 1, e = p + 1;
            while (e < body.length() && depth > 0) {
                char c = body.charAt(e);
                if (c == '[') depth++; else if (c == ']') depth--;
                e++;
            }
            return parseVec(body.substring(p + 1, e - 1));
        }
    }

    // =========================================================================
    //  TEXT CHUNKER  — splits long docs into overlapping word chunks
    // =========================================================================

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0) return new ArrayList<>();
        if (words.length <= chunkWords) return Collections.singletonList(text);
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) { if (j > i) sb.append(' '); sb.append(words[j]); }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }

    // =========================================================================
    //  JSON / PARSE HELPERS
    // =========================================================================

    static String escJson(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                default:   sb.append(c);
            }
        }
        return sb.toString();
    }

    static String jS(String s) { return "\"" + escJson(s) + "\""; }

    static String jVec(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return sb.append(']').toString();
    }

    static float[] parseVec(String s) {
        if (s == null || s.isBlank()) return new float[0];
        List<Float> r = new ArrayList<>();
        for (String p : s.split(","))
            try { r.add(Float.parseFloat(p.trim())); } catch (Exception ignored) {}
        float[] arr = new float[r.size()];
        for (int i = 0; i < r.size(); i++) arr[i] = r.get(i);
        return arr;
    }

    static String extractStr(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"':  result.append('"');  break;
                    case '\\': result.append('\\'); break;
                    case 'n':  result.append('\n'); break;
                    case 'r':  result.append('\r'); break;
                    case 't':  result.append('\t'); break;
                    default:   result.append(body.charAt(p));
                }
            } else { result.append(c); }
            p++;
        }
        return result.toString();
    }

    static int extractInt(String body, String key, int def) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        int end = p;
        while (end < body.length() && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-')) end++;
        try { return Integer.parseInt(body.substring(p, end)); } catch (Exception e) { return def; }
    }

    static float[] parseVecFromBody(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) return new float[0];
        p = body.indexOf('[', p);
        if (p < 0) return new float[0];
        int e = body.indexOf(']', p);
        if (e < 0) return new float[0];
        return parseVec(body.substring(p + 1, e));
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> map = new HashMap<>();
        if (query == null || query.isEmpty()) return map;
        for (String pair : query.split("&")) {
            String[] kv = pair.split("=", 2);
            if (kv.length == 2)
                try {
                    map.put(URLDecoder.decode(kv[0], "UTF-8"),
                            URLDecoder.decode(kv[1], "UTF-8"));
                } catch (Exception ignored) {}
        }
        return map;
    }

    // =========================================================================
    //  HTTP HELPERS
    // =========================================================================

    static void setCors(HttpExchange ex) {
        Headers h = ex.getResponseHeaders();
        h.set("Access-Control-Allow-Origin",  "*");
        h.set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        h.set("Access-Control-Allow-Headers", "Content-Type");
    }

    static void sendJson(HttpExchange ex, String json) throws IOException {
        setCors(ex);
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    // =========================================================================
    //  DEMO DATA  — 20 pre-loaded 16D semantic vectors (4 categories)
    // =========================================================================

    static void loadDemo(VectorDB db) {
        DistFn d = getDistFn("cosine");
        // Dimensions 0-3: CS | 4-7: Math | 8-11: Food | 12-15: Sports
        db.insert("Linked List: nodes connected by pointers", "cs",
                new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, d);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
                new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, d);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
                new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, d);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
                new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, d);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
                new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, d);
        db.insert("Calculus: derivatives integrals and limits", "math",
                new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, d);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
                new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, d);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
                new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, d);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
                new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, d);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
                new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, d);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, d);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
                new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, d);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, d);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, d);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
                new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, d);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
                new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, d);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
                new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, d);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, d);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
                new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, d);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, d);
    }

    // =========================================================================
    //  MAIN — HTTP server + all API endpoints
    // =========================================================================

    public static void main(String[] args) throws Exception {

        VectorDB     db     = new VectorDB(DIMS);
        DocumentDB   docDB  = new DocumentDB();
        OllamaClient ollama = new OllamaClient();

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:" + PORT);
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp
                ? "ONLINE  embed=" + ollama.embedModel + "  gen=" + ollama.genModel
                : "OFFLINE (install from https://ollama.com)"));

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);

        server.createContext("/", ex -> {
            try {
                setCors(ex);
                String method = ex.getRequestMethod();
                String path   = ex.getRequestURI().getPath();
                Map<String, String> params = parseQuery(ex.getRequestURI().getQuery());

                // ── CORS preflight ────────────────────────────────────────────
                if ("OPTIONS".equals(method)) {
                    ex.sendResponseHeaders(204, -1); ex.close(); return;
                }

                // ── Serve index.html ──────────────────────────────────────────
                if ("GET".equals(method) && "/".equals(path)) {
                    File f = new File("index.html");
                    if (!f.exists()) {
                        byte[] msg = "index.html not found in current directory".getBytes();
                        ex.sendResponseHeaders(404, msg.length);
                        ex.getResponseBody().write(msg); ex.close(); return;
                    }
                    byte[] html = Files.readAllBytes(f.toPath());
                    ex.getResponseHeaders().set("Content-Type", "text/html");
                    ex.sendResponseHeaders(200, html.length);
                    try (OutputStream os = ex.getResponseBody()) { os.write(html); }
                    return;
                }

                // ── GET /search ───────────────────────────────────────────────
                if ("GET".equals(method) && "/search".equals(path)) {
                    float[] q = parseVec(params.get("v"));
                    if (q.length != DIMS) {
                        sendJson(ex, "{\"error\":\"need " + DIMS + "D vector\"}"); return;
                    }
                    int    k      = toInt(params.get("k"), 5);
                    String metric = params.getOrDefault("metric", "cosine");
                    String algo   = params.getOrDefault("algo",   "hnsw");
                    VectorDB.SearchOut out = db.search(q, k, metric, algo);
                    StringBuilder ss = new StringBuilder("{\"results\":[");
                    for (int i = 0; i < out.hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        VectorDB.Hit h = out.hits.get(i);
                        ss.append("{\"id\":").append(h.id)
                                .append(",\"metadata\":").append(jS(h.meta))
                                .append(",\"category\":").append(jS(h.cat))
                                .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                                .append(",\"embedding\":").append(jVec(h.emb)).append('}');
                    }
                    ss.append("],\"latencyUs\":").append(out.us)
                            .append(",\"algo\":").append(jS(out.algo))
                            .append(",\"metric\":").append(jS(out.metric)).append('}');
                    sendJson(ex, ss.toString()); return;
                }

                // ── POST /insert ──────────────────────────────────────────────
                if ("POST".equals(method) && "/insert".equals(path)) {
                    String body = readBody(ex);
                    String meta = extractStr(body, "metadata");
                    String cat  = extractStr(body, "category");
                    float[] emb = parseVecFromBody(body, "embedding");
                    if (meta.isEmpty() || emb.length != DIMS) {
                        sendJson(ex, "{\"error\":\"invalid body\"}"); return;
                    }
                    int id = db.insert(meta, cat, emb, getDistFn("cosine"));
                    sendJson(ex, "{\"id\":" + id + "}"); return;
                }

                // ── DELETE /delete/:id ────────────────────────────────────────
                if ("DELETE".equals(method) && path.startsWith("/delete/")) {
                    int id = Integer.parseInt(path.substring("/delete/".length()));
                    sendJson(ex, "{\"ok\":" + db.remove(id) + "}"); return;
                }

                // ── GET /items ────────────────────────────────────────────────
                if ("GET".equals(method) && "/items".equals(path)) {
                    List<VectorItem> items = db.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < items.size(); i++) {
                        if (i > 0) ss.append(',');
                        VectorItem v = items.get(i);
                        ss.append("{\"id\":").append(v.id)
                                .append(",\"metadata\":").append(jS(v.metadata))
                                .append(",\"category\":").append(jS(v.category))
                                .append(",\"embedding\":").append(jVec(v.emb)).append('}');
                    }
                    sendJson(ex, ss.append(']').toString()); return;
                }

                // ── GET /benchmark ────────────────────────────────────────────
                if ("GET".equals(method) && "/benchmark".equals(path)) {
                    float[] q = parseVec(params.get("v"));
                    if (q.length != DIMS) {
                        sendJson(ex, "{\"error\":\"need " + DIMS + "D vector\"}"); return;
                    }
                    long[] b = db.benchmark(q, toInt(params.get("k"), 5),
                            params.getOrDefault("metric", "cosine"));
                    sendJson(ex, "{\"bruteforceUs\":" + b[0] + ",\"kdtreeUs\":" + b[1]
                            + ",\"hnswUs\":" + b[2] + ",\"itemCount\":" + b[3] + "}"); return;
                }

                // ── GET /hnsw-info ────────────────────────────────────────────
                if ("GET".equals(method) && "/hnsw-info".equals(path)) {
                    Map<String, Object> gi = db.hnswInfo();
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"topLayer\":").append(gi.get("topLayer"))
                            .append(",\"nodeCount\":").append(gi.get("nodeCount"))
                            .append(",\"nodesPerLayer\":[");
                    { int[] arr = (int[]) gi.get("nodesPerLayer"); for (int i = 0; i < arr.length; i++) { if (i > 0) ss.append(','); ss.append(arr[i]); } }
                    ss.append("],\"edgesPerLayer\":[");
                    { int[] arr = (int[]) gi.get("edgesPerLayer"); for (int i = 0; i < arr.length; i++) { if (i > 0) ss.append(','); ss.append(arr[i]); } }
                    ss.append("],\"nodes\":[");
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> nodes = (List<Map<String, Object>>) gi.get("nodes");
                    for (int i = 0; i < nodes.size(); i++) {
                        if (i > 0) ss.append(',');
                        Map<String, Object> n = nodes.get(i);
                        ss.append("{\"id\":").append(n.get("id"))
                                .append(",\"metadata\":").append(jS((String) n.get("metadata")))
                                .append(",\"category\":").append(jS((String) n.get("category")))
                                .append(",\"maxLyr\":").append(n.get("maxLyr")).append('}');
                    }
                    ss.append("],\"edges\":[");
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> edges = (List<Map<String, Object>>) gi.get("edges");
                    for (int i = 0; i < edges.size(); i++) {
                        if (i > 0) ss.append(',');
                        Map<String, Object> e = edges.get(i);
                        ss.append("{\"src\":").append(e.get("src"))
                                .append(",\"dst\":").append(e.get("dst"))
                                .append(",\"lyr\":").append(e.get("lyr")).append('}');
                    }
                    sendJson(ex, ss.append("]}").toString()); return;
                }

                // ── POST /doc/insert ──────────────────────────────────────────
                if ("POST".equals(method) && "/doc/insert".equals(path)) {
                    String body  = readBody(ex);
                    String title = extractStr(body, "title");
                    String text  = extractStr(body, "text");
                    if (title.isEmpty() || text.isEmpty()) {
                        sendJson(ex, "{\"error\":\"need title and text\"}"); return;
                    }
                    List<String> chunks = chunkText(text, 250, 30);
                    List<Integer> ids = new ArrayList<>();
                    for (int i = 0; i < chunks.size(); i++) {
                        float[] emb = ollama.embed(chunks.get(i));
                        if (emb.length == 0) {
                            sendJson(ex, "{\"error\":\"Ollama unavailable. "
                                    + "Install from https://ollama.com then run: "
                                    + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                            return;
                        }
                        String ct = (chunks.size() > 1)
                                ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                                : title;
                        ids.add(docDB.insert(ct, chunks.get(i), emb));
                    }
                    StringBuilder ss = new StringBuilder("{\"ids\":[");
                    for (int i = 0; i < ids.size(); i++) { if (i > 0) ss.append(','); ss.append(ids.get(i)); }
                    ss.append("],\"chunks\":").append(chunks.size())
                            .append(",\"dims\":").append(docDB.getDims()).append('}');
                    sendJson(ex, ss.toString()); return;
                }

                // ── GET /doc/list ─────────────────────────────────────────────
                if ("GET".equals(method) && "/doc/list".equals(path)) {
                    List<DocItem> docs = docDB.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < docs.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocItem doc = docs.get(i);
                        String preview = doc.text.length() > 120
                                ? doc.text.substring(0, 120) + "…" : doc.text;
                        ss.append("{\"id\":").append(doc.id)
                                .append(",\"title\":").append(jS(doc.title))
                                .append(",\"preview\":").append(jS(preview))
                                .append(",\"words\":").append(doc.text.split("\\s+").length)
                                .append('}');
                    }
                    sendJson(ex, ss.append(']').toString()); return;
                }

                // ── DELETE /doc/delete/:id ────────────────────────────────────
                if ("DELETE".equals(method) && path.startsWith("/doc/delete/")) {
                    int id = Integer.parseInt(path.substring("/doc/delete/".length()));
                    sendJson(ex, "{\"ok\":" + docDB.remove(id) + "}"); return;
                }

                // ── POST /doc/search ──────────────────────────────────────────
                if ("POST".equals(method) && "/doc/search".equals(path)) {
                    String body = readBody(ex);
                    String q2   = extractStr(body, "question");
                    int k2      = extractInt(body, "k", 3);
                    if (q2.isEmpty()) { sendJson(ex, "{\"error\":\"need question\"}"); return; }
                    float[] qEmb = ollama.embed(q2);
                    if (qEmb.length == 0) { sendJson(ex, "{\"error\":\"Ollama unavailable\"}"); return; }
                    List<float[]> hits = docDB.search(qEmb, k2);
                    StringBuilder ss = new StringBuilder("{\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        int id = (int) hits.get(i)[1]; DocItem doc = docDB.get(id);
                        if (doc != null)
                            ss.append("{\"id\":").append(id)
                                    .append(",\"title\":").append(jS(doc.title))
                                    .append(",\"distance\":").append(String.format(Locale.US,"%.4f",hits.get(i)[0]))
                                    .append('}');
                    }
                    sendJson(ex, ss.append("]}").toString()); return;
                }

                // ── POST /doc/ask (RAG) ───────────────────────────────────────
                if ("POST".equals(method) && "/doc/ask".equals(path)) {
                    String body = readBody(ex);
                    String q2   = extractStr(body, "question");
                    int k2      = extractInt(body, "k", 3);
                    if (q2.isEmpty()) { sendJson(ex, "{\"error\":\"need question\"}"); return; }
                    float[] qEmb = ollama.embed(q2);
                    if (qEmb.length == 0) { sendJson(ex, "{\"error\":\"Ollama unavailable\"}"); return; }
                    List<float[]> hits = docDB.search(qEmb, k2);
                    StringBuilder ctx = new StringBuilder();
                    for (int i = 0; i < hits.size(); i++) {
                        DocItem doc = docDB.get((int) hits.get(i)[1]);
                        if (doc != null)
                            ctx.append("[").append(i + 1).append("] ")
                                    .append(doc.title).append(":\n").append(doc.text).append("\n\n");
                    }
                    String prompt =
                            "You are a helpful assistant. Answer the user's question directly. "
                                    + "Use the provided context if it contains relevant information. "
                                    + "If it doesn't, just use your own general knowledge. "
                                    + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things "
                                    + "like 'the context doesn't mention'. Just answer the question naturally.\n\n"
                                    + "Context:\n" + ctx
                                    + "Question: " + q2 + "\n\nAnswer:";
                    String answer = ollama.generate(prompt);
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"answer\":").append(jS(answer))
                            .append(",\"model\":").append(jS(ollama.genModel))
                            .append(",\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        int id = (int) hits.get(i)[1]; DocItem doc = docDB.get(id);
                        if (doc != null)
                            ss.append("{\"id\":").append(id)
                                    .append(",\"title\":").append(jS(doc.title))
                                    .append(",\"text\":").append(jS(doc.text))
                                    .append(",\"distance\":").append(String.format(Locale.US,"%.4f",hits.get(i)[0]))
                                    .append('}');
                    }
                    ss.append("],\"docCount\":").append(docDB.size()).append('}');
                    sendJson(ex, ss.toString()); return;
                }

                // ── GET /status ───────────────────────────────────────────────
                if ("GET".equals(method) && "/status".equals(path)) {
                    boolean up = ollama.isAvailable();
                    sendJson(ex, "{\"ollamaAvailable\":" + up
                            + ",\"embedModel\":"  + jS(ollama.embedModel)
                            + ",\"genModel\":"    + jS(ollama.genModel)
                            + ",\"docCount\":"    + docDB.size()
                            + ",\"docDims\":"     + docDB.getDims()
                            + ",\"demoDims\":"    + DIMS
                            + ",\"demoCount\":"   + db.size() + "}");
                    return;
                }

                // ── GET /stats ────────────────────────────────────────────────
                if ("GET".equals(method) && "/stats".equals(path)) {
                    sendJson(ex, "{\"count\":" + db.size()
                            + ",\"dims\":"       + DIMS
                            + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                            + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
                    return;
                }

                // 404
                byte[] msg = "Not Found".getBytes();
                ex.sendResponseHeaders(404, msg.length);
                ex.getResponseBody().write(msg); ex.close();

            } catch (Exception err) {
                System.err.println("Handler error on " + ex.getRequestURI() + ": " + err.getMessage());
                try { ex.sendResponseHeaders(500, -1); ex.close(); } catch (Exception ignored) {}
            }
        });

        server.setExecutor(Executors.newFixedThreadPool(8));
        server.start();
        System.out.println("Server started → http://localhost:" + PORT);
    }

    private static int toInt(String s, int def) {
        try { return (s == null || s.isBlank()) ? def : Integer.parseInt(s.trim()); }
        catch (Exception e) { return def; }
    }
}