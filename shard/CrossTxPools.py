from blockchain.tx import Tx


class CrossTxPools(object):
    def __init__(self):
        self.OriginalTxsQueue={}

    def init(self):
        self.OriginalTxsQueue={}

    def addOriginalTx(self, tx: Tx) -> None:
        if tx.OrigTxHash in self.OriginalTxsQueue:
            raise ValueError(f'Duplicate cross-shard original: {tx.OrigTxHash}')
        self.OriginalTxsQueue[tx.OrigTxHash] = tx

    def getOriginalTx(self, OriginalTxHash):
        return self.OriginalTxsQueue.get(OriginalTxHash)

    def removeOriginalTx(self, OriginalTxHash) -> None:
        if OriginalTxHash in self.OriginalTxsQueue:
           del self.OriginalTxsQueue[OriginalTxHash]
