from typing import List

from blockchain.tx import Tx


class Txpools(object):
    def __init__(self, max_size = 100):
        self.TxsQueue = {}
        self.max_size = max_size
        self.load = 0

    def __len__(self):
        return len(self.TxsQueue)

    def init(self):
        self.TxsQueue = {}
        self.load = 0

    def addTx(self,Hash, originalHash,tx: Tx,test = False) -> None:
        self.load += int((Hash,originalHash) not in self.TxsQueue)
        self.TxsQueue[(Hash,originalHash)] = tx

    def addTxs(self, txs: list,originalHashs) -> None:

        for tx in txs:

            self.addTx(tx.Hash,tx.OrigTxHash,tx)


    def get(self, txHash: str,originalHash):
        return self.TxsQueue.get((txHash,originalHash))

    def getAllTxs(self) -> List[Tx]:
        return list(self.TxsQueue.values())

    def getEnoughToFillblock(self, TxIsProcessed, sid):
        # Reading/selecting candidates must never mark transactions confirmed.
        return [tx for tx in self.TxsQueue.values()
                if tx.OrigTxHash or self.isPending(tx, TxIsProcessed)]

    def delTx(self, tx,Hash,originalHash,sid):
        self.removeTx(tx.Hash,tx.OrigTxHash)

    def getEnoughToFillblockv1(self, blockSize, TxIsProcessed, sid):
        return self.getEnoughToFillblock(TxIsProcessed, sid)[:max(0, int(blockSize))]


    def isPending(self, tx, TxIsProcessed):
        return all(TxIsProcessed.get(inp.SenptHash, False) for inp in tx.TxIns)


    def removeTx(self, txHash: str,originalHash):
        del self.TxsQueue[(txHash,originalHash)]
        self.load -= 1

    def popTx(self, txHash, originalHash):
        tx = self.TxsQueue.pop((txHash, originalHash), None)
        self.load -= int(tx is not None)
        return tx

    def popAllTxs(self) -> List[Tx]:
        txs = list(self.TxsQueue.values())
        self.TxsQueue = {}
        self.load = 0
        return txs

    def processBlock(self, txs: List[Tx],originalHashs) -> None:
        for tx,originalHash in zip(txs,originalHashs):
            self.removeTx(tx.Hash,originalHash)

