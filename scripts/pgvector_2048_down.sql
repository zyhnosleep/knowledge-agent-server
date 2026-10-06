-- 把 pgvector_2048_up.sql 回退掉:2048 维索引换回 2560 维。
--
-- 前提:先把 .env 里的 EMBEDDING_DIMENSIONS / OLLAMA_EMBEDDING_DIMENSIONS
-- 改回 2560,并把 OLLAMA_EMBEDDING_BASE_URL 从本地服务指回原来的 embedder,
-- 否则写进去的还是 2048 维向量,存进 vector(2560) 会直接报错。
--
-- 会丢掉 2048 表里的全部向量(也就是这次多模态实验的索引)——2560 维那张
-- 备份表原样恢复,它才是回退要保住的东西。

BEGIN;

-- 1) 丢掉 2048 的新表(它的向量可以重建,不需要留)
DROP TABLE IF EXISTS public.document_chunk_pgvector_index;

-- 2) 备份表改回原名
ALTER TABLE public.document_chunk_pgvector_index_v2560_backup
    RENAME TO document_chunk_pgvector_index;

ALTER INDEX public.document_chunk_pgvector_index_v2560_backup_pkey
    RENAME TO document_chunk_pgvector_index_pkey;
ALTER INDEX public.ix_document_chunk_pgvector_index_v2560_backup_document_id
    RENAME TO ix_document_chunk_pgvector_index_document_id;
-- 注意:下面这个名字看着少了一截,不是笔误。PostgreSQL 的标识符上限是 63
-- 字符,上行时这个名字被自动截断成了 ..._document_parse_ve。写全名会报
-- "index does not exist"。
ALTER INDEX public.ix_document_chunk_pgvector_index_v2560_backup_document_parse_ve
    RENAME TO ix_document_chunk_pgvector_index_document_parse_version;

ALTER TABLE public.document_chunk_pgvector_index
    RENAME CONSTRAINT document_chunk_pgvector_index_v2560_backup_chunk_id_fkey
    TO document_chunk_pgvector_index_chunk_id_fkey;
ALTER TABLE public.document_chunk_pgvector_index
    RENAME CONSTRAINT document_chunk_pgvector_index_v2560_backup_document_id_fkey
    TO document_chunk_pgvector_index_document_id_fkey;

COMMIT;
