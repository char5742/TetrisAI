import CodecZstd
import Serialization

function serialize(data)
    buffer = IOBuffer()
    Serialization.serialize(buffer, data)
    compressed = Config.compress ? transcode(CodecZstd.ZstdCompressor, take!(buffer)) : take!(buffer)
    close(buffer)
    compressed
end

function deserialize(byte)
    buffer = IOBuffer(byte)
    stream = Config.compress ? CodecZstd.ZstdDecompressorStream(buffer) : buffer
    decompressed  = Serialization.deserialize(stream)
    close(buffer)
    close(stream)
    decompressed
end