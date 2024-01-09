import CodecZstd
import Serialization

function serialize(data)
    buffer = IOBuffer()
    Serialization.serialize(buffer, data)
    Config.compress ? transcode(CodecZstd.ZstdCompressor, take!(buffer)) : take!(buffer)
end

function deserialize(byte)
    buffer = IOBuffer(byte)
    Serialization.deserialize(Config.compress ? CodecZstd.ZstdDecompressorStream(buffer) : buffer)
end