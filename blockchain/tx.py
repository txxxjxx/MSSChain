import hashlib
import struct

import blockchain.params as p





def byte_slice_append(*b: bytes) -> bytes:
    tmp = bytearray()
    for item in b:

        tmp.extend(item)
    return bytes(tmp)

def float_to_byte(f: float) -> bytes:
    return struct.pack('<d', f)
def uint_to_byte(u: int) -> bytes:

    return struct.pack('<Q', u)

def hash(msg: bytes) -> bytes:
    hashed_msg = hashlib.sha256(msg).digest()
    if len(hashed_msg) != 32:
        raise ValueError("Hash not 32 bytes")
    return hashed_msg



class TxIn(object):
    __slots__ = ('SenptHash','InputAddr','InputVout','position','Sid')

    def __init__(self, SenptHash,InputHash,position, InputVout,sid):
        self.SenptHash = SenptHash
        self.InputAddr = InputHash
        self.InputVout = InputVout
        self.position = position
        self.Sid = sid

    def bytes_except_sig(self) -> bytes:
        return byte_slice_append(self.SenptHash.encode('utf-8'), b'\0',
                                 self.InputAddr.encode('utf-8'), b'\0',
                                 uint_to_byte(self.position), uint_to_byte(self.Sid))

class TxOut(object):
    __slots__ = ('Sid','Addr','position','Value')

    def __init__(self, Addr,position, Value, Sid):
        self.Sid = Sid
        self.Addr = Addr
        self.position = position
        self.Value = Value

    def bytes(self) -> bytes:
        return byte_slice_append(self.Addr.encode('utf-8'), b'\0',
                                 uint_to_byte(self.position), float_to_byte(self.Value),
                                 uint_to_byte(self.Sid))



class Tx(object):
    __slots__ = ('is_create','Hash','is_coinbase','Size','TxInCount','TxOutCount',
                 'TxInValue','TxOutValue','TxIns','TxOuts','Height','TimeStamp',
                 'UTXOed','Sid','TD','OrigTxHash','tx_consensus')

    def __init__(self, hash, is_coinbase, size, txincount, txoutcount, txinvalue, txoutvalue,txins, txouts, height:int,time):
        self.is_create = False
        self.Hash = hash
        self.is_coinbase = is_coinbase
        self.Size = size
        self.TxInCount = txincount
        self.TxOutCount = txoutcount
        self.TxInValue = txinvalue
        self.TxOutValue = txoutvalue
        self.TxIns = txins
        self.TxOuts = txouts
        self.Height = height
        self.TimeStamp = time
        self.UTXOed = False
        self.Sid = 0
        # The main simulator stores compact float32 TD vectors in its own
        # cache.  Legacy graph code fills this field only when it is used.
        self.TD = None
        self.OrigTxHash = ""
        self.tx_consensus = False


    def string(self):
        tx_info = [
            f"Hash: {self.Hash}",
            f"Coinbase: {self.is_coinbase}",
            f"Size: {self.Size} bytes",
            f"Inputs: {self.TxInCount} (Total: {self.TxInValue})",

            f"Outputs: {self.TxOutCount} (Total: {self.TxOutValue})",
            f"Height: {self.Height}",
            f"Timestamp: {self.TimeStamp}",
            f"Shard ID (Sid): {self.Sid}",
            f"UTXOed: {self.UTXOed}",
            f"Original Tx Hash: {self.OrigTxHash or 'N/A'}",
            f"TD (Transaction Distribution): {self.TD}"
        ]
        print("\n".join(tx_info))

    def setHash(self):
        self.Hash = self.calculateHash()

    def calculateHash(self):
        b = bytearray()
        for input in self.TxIns or []:
            b.extend(input.bytes_except_sig())

        for output in self.TxOuts or []:
            b.extend(output.bytes())
        return hash(byte_slice_append(b,self.OrigTxHash.encode('utf-8')))
