
struct PositionalEncodingLayer <: Lux.LuxCore.AbstractExplicitLayer
    features::Int
    seq_len::Int
end

Lux.initialparameters(rng::AbstractRNG, layer::PositionalEncodingLayer) = NamedTuple()

function Lux.initialstates(rng::AbstractRNG, layer::PositionalEncodingLayer)
    features = layer.features
    seq_len = layer.seq_len
    pos_enc = [
        i % 2 == 1 ? sin(j / 10000^(2 * i / features)) : cos(j / 10000^(2 * i / features))
        for i in 1:features, j in 1:seq_len, _ in 1:1
    ]
    (pos_enc=pos_enc,)
end
Lux.parameterlength(l::PositionalEncodingLayer) = 0
Lux.statelength(l::PositionalEncodingLayer) = l.features * l.seq_len

function (::PositionalEncodingLayer)(x, ps, st)
    return x .+ st.pos_enc, st
end

struct MultiHeadAttention <: Lux.LuxCore.AbstractExplicitContainerLayer{(:query, :key, :value, :dropout, :output)}
    query
    key
    value
    dropout
    output
    nheads
end

function (m::MultiHeadAttention)((x, encoder_output, mask), ps, st)
    # x = (features, seq_len, batch_size) from decoder
    # encoder_output = (features, seq_len, batch_size) from encoder
    q, _ = m.query(x, ps.query, st.query)

    # Use encoder output as key and value
    k, _ = m.key(encoder_output, ps.key, st.key)
    v, _ = m.value(encoder_output, ps.value, st.value)

    st_dropout = nothing
    function fdrop(x)
        x, st_dropout = m.dropout(x, ps.dropout, st.dropout)
        x
    end
    values, _ = dot_product_attention(q, k, v; mask=mask, fdrop=fdrop, nheads=m.nheads)

    output, _ = m.output(values, ps.output, st.output)
    st = merge(st, (dropout=st_dropout,))
    return output, st
end

struct DecoderBlock <: Lux.LuxCore.AbstractExplicitContainerLayer{(:mha, :layer_norm1, :ffn, :dropout, :layer_norm2)}
    mha
    layer_norm1
    ffn
    dropout
    layer_norm2
end

function (m::DecoderBlock)((x, encoder_output, mask), ps, st)
    z = x
    norm_out, st_layer_norm1 = m.layer_norm1(z, ps.layer_norm1, st.layer_norm1)
    attn_output, st_mha = m.mha((norm_out, encoder_output, mask), ps.mha, st.mha)
    z = attn_output + z

    norm_out, st_layer_norm2 = m.layer_norm2(z, ps.layer_norm2, st.layer_norm2)
    ffn_output, _ = m.ffn(norm_out, ps.ffn, st.ffn)
    dropout_out, st_dropout = m.dropout(ffn_output, ps.dropout, st.dropout)
    z = dropout_out + z

    st = merge(st, (mha=st_mha, layer_norm1=st_layer_norm1, layer_norm2=st_layer_norm2, dropout=st_dropout,))
    return (z, encoder_output, mask), st
end


struct Decoder <: Lux.LuxCore.AbstractExplicitContainerLayer{(:q_embedding, :q_positional_encoding, :kv_embedding, :kv_positional_encoding, :blocks, :output)}
    q_embedding
    q_positional_encoding
    kv_embedding
    kv_positional_encoding
    blocks
    output
end

function (m::Decoder)((q, kv, mask), ps, st)
    # x = (token_size, seq_len, batch_size)
    kv, _ = m.kv_embedding(kv, ps.kv_embedding, st.kv_embedding)
    kv, _ = m.kv_positional_encoding(kv, ps.kv_positional_encoding, st.kv_positional_encoding)
    q, _ = m.q_embedding(q, ps.q_embedding, st.q_embedding)
    q, _ = m.q_positional_encoding(q, ps.q_positional_encoding, st.q_positional_encoding)
    # x = (features, seq_len, batch_size)

    (x, _, _), st_blocks = m.blocks((q, kv, mask), ps.blocks, st.blocks)
    output, st_out = m.output(x, ps.output, st.output)
    # output = (token_size, seq_len, batch_size)
    st = merge(st, (blocks=st_blocks, output=st_out))
    return output, st
end
