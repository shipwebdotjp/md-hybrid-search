# Decisions Log

## 2026-05-25
- embedding dimension は `embedding_dim` / `dim` を優先し、無い場合は最初の embedding で観測した次元を検証用に保持する
- embedding cache は持たず、毎回 embedder を呼ぶ

## 2026-05-24
- v0からrerankerを受け取る実装に変更
