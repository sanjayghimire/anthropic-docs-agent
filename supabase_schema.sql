-- Enable the vector extension
create extension if not exists vector;

-- Create documents table
create table if not exists documents (
    id          bigserial primary key,
    content     text not null,
    url         text not null,
    title       text,
    chunk_index integer default 0,
    embedding   vector(1536),
    created_at  timestamp with time zone default now()
);

-- Prevent duplicates on re-run
create unique index if not exists documents_url_chunk_idx 
    on documents(url, chunk_index);

-- Search function
create or replace function match_documents(
    query_embedding  vector(1536),
    match_count      int default 5
)
returns table (
    id          bigint,
    content     text,
    url         text,
    title       text,
    similarity  float
)
language sql stable
as $$
    select
        id,
        content,
        url,
        title,
        1 - (embedding <=> query_embedding) as similarity
    from documents
    order by embedding <=> query_embedding
    limit match_count;
$$;