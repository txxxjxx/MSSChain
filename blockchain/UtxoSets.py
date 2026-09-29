from typing import Dict, Tuple

from blockchain.tx import TxOut


class UTXOSets(object):
    def __init__(self):
        self.sets: Dict[str, Dict[Tuple[str,int],TxOut]] = {}

    def init(self):
        self.sets: Dict[str, Dict[Tuple[str,int],TxOut]] = {}

    def add(self,hash, Addr,position,oTx: TxOut):
        if hash not in self.sets:
            self.sets.setdefault(hash, {})[(Addr,position)] = oTx
        else:
            self.sets[hash][(Addr,position)] = oTx

    def removeOutput(self, Hash, oTxAddr, position):
        del self.sets[Hash][(oTxAddr, position)]
        if not self.sets[Hash]:
            del self.sets[Hash]

    def getOutput(self,Hahs, oTxAddr,position):
        if Hahs not in self.sets:


            return None
        if (oTxAddr,position) not in self.sets[Hahs]:
            return None

        return self.sets[Hahs][(oTxAddr,position)]

    def getAndRemoveOutput(self, Hash, oTxAddr, position):
        output = self.getOutput(Hash, oTxAddr, position)
        if output is not None:
            self.removeOutput(Hash, oTxAddr, position)
        return output


