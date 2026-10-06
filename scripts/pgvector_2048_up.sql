-- 把 chunk 向量索引从 2560 维换到 2048 维。
--
-- 为什么要换:文字索引改用 Qwen3-VL-Embedding-2B(2048 维),好让文字块和
-- 图块落在同一个向量空间里,一次检索就能同时排出两类结果。
--
-- 为什么要改名而不是直接改列:pgvector 的 vector(2560) 是列级约束,装不下
-- 2048 维;而现有的 363 个 2560 维向量要留着做回退。所以旧表整体改名留档,
-- 新建一张同名的 2048 维表——同名的原因是 PGVectorStore._TABLE_NAME 写死了
-- 这个名字,保持它,现有检索代码一行都不用改。
--
-- 注意:PostgreSQL 的索引名是全 schema 唯一的,所以旧表的索引/约束必须
-- 一并改名,否则新表建同名索引会冲突。另外标识符上限 63 字符,那条最长的
-- 索引名会被自动截断成 ..._document_parse_ve——回退脚本里写的就是截断后的
-- 名字,别以为它少打了一截。

BEGIN;

-- 1) 旧表改名留档(数据一行不动)
ALTER TABLE public.document_chunk_pgvector_index
    RENAME TO document_chunk_pgvector_index_v2560_backup;

ALTER INDEX public.document_chunk_pgvector_index_pkey
    RENAME TO document_chunk_pgvector_index_v2560_backup_pkey;
ALTER INDEX public.ix_document_chunk_pgvector_index_document_id
    RENAME TO ix_document_chunk_pgvector_index_v2560_backup_document_id;
ALTER INDEX public.ix_document_chunk_pgvector_index_document_parse_version
    RENAME TO ix_document_chunk_pgvector_index_v2560_backup_document_parse_version;

ALTER TABLE public.document_chunk_pgvector_index_v2560_backup
    RENAME CONSTRAINT document_chunk_pgvector_index_chunk_id_fkey
    TO document_chunk_pgvector_index_v2560_backup_chunk_id_fkey;
ALTER TABLE public.document_chunk_pgvector_index_v2560_backup
    RENAME CONSTRAINT document_chunk_pgvector_index_document_id_fkey
    TO document_chunk_pgvector_index_v2560_backup_document_id_fkey;

-- 2) 新建 2048 维的同名表,结构照抄旧表
CREATE TABLE public.document_chunk_pgvector_index (
    chunk_id character varying(36) NOT NULL,
    document_id character varying(36) NOT NULL,
    embedding public.vector(2048) NOT NULL,
    parse_version character varying(128) DEFAULT 'legacy'::character varying NOT NULL
);
ALTER TABLE public.document_chunk_pgvector_index OWNER TO root;

ALTER TABLE ONLY public.document_chunk_pgvector_index
    ADD CONSTRAINT document_chunk_pgvector_index_pkey PRIMARY KEY (chunk_id);

CREATE INDEX ix_document_chunk_pgvector_index_document_id
    ON public.document_chunk_pgvector_index USING btree (document_id);
CREATE INDEX ix_document_chunk_pgvector_index_document_parse_version
    ON public.document_chunk_pgvector_index USING btree (document_id, parse_version);

ALTER TABLE ONLY public.document_chunk_pgvector_index
    ADD CONSTRAINT document_chunk_pgvector_index_chunk_id_fkey
    FOREIGN KEY (chunk_id) REFERENCES public.document_chunks(id) ON DELETE CASCADE;
ALTER TABLE ONLY public.document_chunk_pgvector_index
    ADD CONSTRAINT document_chunk_pgvector_index_document_id_fkey
    FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE CASCADE;

COMMIT;
