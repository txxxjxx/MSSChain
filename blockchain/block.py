import hashlib
import struct

from blockchain.tx import Tx

def uint_to_byte(u: int) -> bytes:
    return struct.pack('<Q', u)

def byte_slice_append(*b: bytes) -> bytes:
    tmp = bytearray()
    for item in b:
        tmp.extend(item)
    return bytes(tmp)

def hash(msg: bytes) -> bytes:
    hashed_msg = hashlib.sha256(msg).digest()
    if len(hashed_msg) != 32:
        raise ValueError("Hash not 32 bytes")
    return hashed_msg


class Block(object):
    def __init__(self,Hash, PreViousHash, Iteration, ShardID, Transactions):
        self.Hash = Hash
        self.PreViousHash = PreViousHash
        self.Iteration = Iteration
        self.ShardID = ShardID
        self.Transactions = Transactions

    def contentHash(self):
        payload = bytearray(byte_slice_append(uint_to_byte(self.ShardID),
                            bytes(self.PreViousHash),uint_to_byte(self.Iteration)))
        for tx in self.Transactions or []:
            for part in (tx.Hash.encode('utf-8'),tx.OrigTxHash.encode('utf-8'),tx.calculateHash()):
                payload.extend(uint_to_byte(len(part)))
                payload.extend(part)
        return hash(bytes(payload))

    def calculateHash(self):
        self.Hash = self.contentHash()
        return self.Hash

