import copy
import math
from decimal import Decimal
import hashlib
import logging
import struct
from collections import defaultdict
from typing import Dict

from blockchain.UtxoSets import UTXOSets as UTXO
from blockchain import blockchain
from blockchain.block import Block
from blockchain.tx import TxIn
from blockchain.tx import TxOut
from blockchain.tx import Tx
from shard.CrossTxPools import CrossTxPools
from shard.Txpools import Txpools
import blockchain.params as p

logger = logging.getLogger(__name__)

class Shard(object):
    def __init__(self, ID):
        self.shardNum = p.ShardNum
        # 图
        #self.partition = Partition()
        self.blocksize = p.BlockSize
        self.blockchain = None
        self.tx_num = 0
        self.txPools=Txpools()
        self.crossTxPool= CrossTxPools()
        self.utxoSets = UTXO()
        self.EffectiveTransaction = 0
        self.cpu_needs = 0
        self.bw_needs = 0
        self.w = [1, 1]
        self.blocktime = 8
        self.slot_duration = 1.0
        self.transaction_size = 500 * p.BYTE
        self.resource_model = 'paper_eq10'
        self.valid_ratio = 0.8
        self.cpu_tps_per_core = 312.5
        self.tx_list = []
        self.ID = ID
        self.resources = [200, 200]

        self.normal = 0

    def setBlockchain(self, blockchain):
        self.blockchain = blockchain


    def genGenesisBlock(self):
        def byte_slice_append(*b: bytes) -> bytes:
            tmp = bytearray()
            for item in b:
                tmp.extend(item)
            return bytes(tmp)
        def uint_to_byte(u: int) -> bytes:
            return struct.pack('<Q', u)
        random_bytes = hashlib.sha256(b'MSSChain deterministic genesis').digest()
        sid_bytes = uint_to_byte(self.ID)
        combined_bytes = byte_slice_append(random_bytes, sid_bytes)
        txHash = hashlib.sha256(combined_bytes).digest()
        genesisBlock = Block(txHash,bytearray(32),0,self.ID,None)
        genesisBlock.calculateHash()
        blocks = []


        bc = blockchain.BlockChain(self.ID,blocks,bytearray(32),None)
        bc.add(genesisBlock)
        self.setBlockchain(bc)


    def createProposeBlock(self,shardI,TxIsProcessed,TxBelongShard):
        # The placement map lets create_block admit a local child after its
        # parent has already been selected earlier in the same block.
        cpu_need,bw_need,_,txes = self.create_block(
            self.resources,TxIsProcessed,self.ID,placements=TxBelongShard)


        Txes = self.processTransaciton(copy.deepcopy(txes),self.ID,TxBelongShard)
        block = Block(bytes(32),self.blockchain.LatestBlock,shardI,self.ID,Txes)
        block.calculateHash()
        return block,cpu_need,bw_need


    def processTransaciton(self, txlist,sid,TxBelongShard):
        processTxes = []
        spentUTXOSet = UTXO()
        addUTXOSet = UTXO()
        tmpCrossTxPool = CrossTxPools()

        for tx in txlist:

            if tx.is_coinbase:
                self.normal +=1
                i= 0
                for out in tx.TxOuts:
                    addUTXOSet.add(tx.Hash,out.Addr,i,out)
                    i += 1
                processTxes.append(tx)
                continue
            if len(tx.OrigTxHash) == 0 and len(tx.Hash) == 64:

                normal = True
                for inp in tx.TxIns:
                    if inp.SenptHash in TxBelongShard:
                        _,inp_sid = TxBelongShard[inp.SenptHash]
                        if inp_sid != sid:
                            normal = False
                            break
                    else:
                        raise ValueError(f"Unknown parent transaction: {inp.SenptHash}")

                if normal == True:
                    self.normal+=1
                    res = self.processNormalTransaction(tx, spentUTXOSet, addUTXOSet,sid)

                    if res:
                        processTxes.append(tx)
                else:

                    newCrossTxes = self.processTransactionWithUnkowInputs(tx, spentUTXOSet, addUTXOSet,sid,TxBelongShard)
                    #self.cst[sid] += 1
                    if len(newCrossTxes) == 0:
                        raise ValueError("len of new cross-txes was 0")
                    if newCrossTxes != None:
                        if tx.OrigTxHash == "" or tx.Hash != "":
                            raise ValueError("originaltx hashes was not changed")
                        processTxes.append(tx)

                        tmpCrossTxPool.addOriginalTx(tx)

                        processTxes.extend(newCrossTxes)

            elif tx.OrigTxHash != "" and tx.Hash == "":
                ok = self.processIncommingCrossTx(tx, spentUTXOSet, addUTXOSet,sid,TxBelongShard)

                if ok:
                    processTxes.append(tx)
            elif tx.OrigTxHash != "" and tx.Hash != "":

                newTx,ok = self.processCrossTxResponse(tx,spentUTXOSet,addUTXOSet,tmpCrossTxPool,sid,TxBelongShard)
                if ok:
                    processTxes.append(tx)
                    if newTx != None:
                        processTxes.append(newTx)
            else:
                raise ValueError("this shouldnt be reached?")

        return processTxes




    def processCrossTxResponse(self,tx,spentUTXOSet,addUTXOSet,tmpCrossTxpool: CrossTxPools,sid,TxBelongShard):

        if len(tx.TxOuts) != len(tx.TxIns):
            raise ValueError("length of crossTxResponse inputs was not equal to len of outputs")

        original = self.crossTxPool.getOriginalTx(tx.OrigTxHash)
        if original == None:
            original = tmpCrossTxpool.getOriginalTx(tx.OrigTxHash)

            if original == None:
                raise ValueError("no original tx")

        if original.OrigTxHash != tx.OrigTxHash:
            raise ValueError("orighash not equal")

        permitted = {(inp.SenptHash,inp.InputAddr,inp.position) for inp in original.TxIns}
        response_keys = [(inp.SenptHash,inp.InputAddr,inp.position) for inp in tx.TxIns]
        if len(set(response_keys)) != len(response_keys) or not set(response_keys) <= permitted:
            raise ValueError('Cross-shard response contains an unrequested or duplicate outpoint')
        for inp,out in zip(tx.TxIns,tx.TxOuts):
            parent = TxBelongShard[inp.SenptHash][0]
            owner = TxBelongShard[inp.SenptHash][1]
            expected = parent.TxOuts[inp.position]
            actual_tuple = (out.Addr,out.position,Decimal(str(out.Value)))
            expected_tuple = (expected.Addr,expected.position,Decimal(str(expected.Value)))
            if actual_tuple != expected_tuple:
                raise ValueError('Cross-shard response does not prove the referenced output')
            if out.Sid != owner:
                raise ValueError('Cross-shard response reports the wrong source shard')

        for i, out in enumerate(tx.TxOuts):

            addUTXOSet.add(tx.TxIns[i].SenptHash,tx.TxIns[i].InputAddr,tx.TxIns[i].position, out)


        allInputsCovered = True

        for inp in original.TxIns:
            outTx = self.utxoSets.getOutput(inp.SenptHash,inp.InputAddr,inp.position)
            if outTx == None:
                outTx = addUTXOSet.getOutput(inp.SenptHash,inp.InputAddr,inp.position)

                if outTx == None:

                    allInputsCovered = False
                    break

            if spentUTXOSet.getOutput(inp.SenptHash,inp.InputAddr,inp.position) != None:
                allInputsCovered = False

                break

        if allInputsCovered == False:
            return  None, True

        newTx = Tx("", False, original.Size, 0,len(original.TxOuts),None,None,None,original.TxOuts,original.Height,original.TimeStamp)

        newInputs = []
        for inp in original.TxIns:
            outTx = self.utxoSets.getOutput(inp.SenptHash,inp.InputAddr,inp.position)
            if outTx == None:
                outTx = addUTXOSet.getOutput(inp.SenptHash,inp.InputAddr,inp.position)
                if outTx == None:
                    if spentUTXOSet.getOutput(inp.SenptHash,inp.InputAddr,inp.position) == None:
                        raise ValueError("spent")
                    raise ValueError("outTx did not exists")

            if spentUTXOSet.getOutput(inp.SenptHash,inp.InputAddr,inp.position) != None:
                raise ValueError("spent")

            newInp = TxIn(inp.SenptHash, inp.InputAddr,inp.position, inp.InputVout, sid)
            newInputs.append(newInp)
            spentUTXOSet.add(inp.SenptHash,inp.InputAddr,inp.position,outTx)
        newTx.TxIns = newInputs
        newTx.TxInCount = len(newInputs)

        newTx.OrigTxHash = original.OrigTxHash
        newTx.setHash()
        newTx.Hash = newTx.Hash.hex()

        i = 0
        for outTx in original.TxOuts:
            addUTXOSet.add(tx.OrigTxHash,outTx.Addr,i,outTx)
            i += 1

        tmpCrossTxpool.removeOriginalTx(original.OrigTxHash)

        return newTx, True



    def processIncommingCrossTx(self, tx, spentUTXOSet, addUTXOSet, sid,TxBelongShard):
        if tx.TxOuts != None:
            raise ValueError("outputs was not nil in incomming cross-tx")
        if not tx.TxIns or tx.OrigTxHash not in TxBelongShard:
            raise ValueError('Cross-shard request has no admitted original or inputs')
        original = TxBelongShard[tx.OrigTxHash][0]
        permitted = {(inp.SenptHash,inp.InputAddr,inp.position) for inp in original.TxIns}
        request_keys = [(inp.SenptHash,inp.InputAddr,inp.position) for inp in tx.TxIns]
        if (len(set(request_keys)) != len(request_keys) or not set(request_keys) <= permitted
                or any(TxBelongShard[inp.SenptHash][1] != sid for inp in tx.TxIns)):
            raise ValueError('Cross-shard request contains an invalid outpoint or destination')
        _, inp_sid = TxBelongShard[tx.TxIns[0].SenptHash]
        if sid != inp_sid:
            raise ValueError("incomming cross tx not beloning in this committee")


        for inp in tx.TxIns:
            #print(tx.Hash,inp.SenptHash,inp.position)
            if self.validateInput(inp, spentUTXOSet, addUTXOSet, sid) == False:
                raise ValueError("Incoming cross-tx input not valid")


        newOuts = []
        for inp in tx.TxIns:
            out = self.spendInputToNewOutput(tx.OrigTxHash,inp, spentUTXOSet, sid)
            _,out.Sid = TxBelongShard[inp.SenptHash]
            newOuts.append(out)

        if tx.TxOuts != None:
            raise ValueError("outputs was not nil")

        tx.TxOuts = newOuts

        if tx.Hash != "":
            raise ValueError("Hash was not nil")

        tx.TxOutCount = len(tx.TxOuts)
        tx.setHash()
        tx.Hash = tx.Hash.hex()
        tx.tx_consensus = True
        tx.Size = self._protocol_message_size(tx)
        return True


    def spendInputToNewOutput(self, Hash,iTx: TxIn, spentUTXOSet :UTXO,sid):
        outTx = self.utxoSets.getOutput(iTx.SenptHash,iTx.InputAddr,iTx.position)
        if outTx is None:
            raise ValueError("outTx is None")
        spentUTXOSet.add(iTx.SenptHash, iTx.InputAddr, iTx.position, outTx)
        # Relay metadata must never mutate the canonical UTXO object, which
        # may also be referenced by an already-hashed historical block.
        return copy.deepcopy(outTx)


    def processTransactionWithUnkowInputs(self,tx, spentUTXOSet, addUTXOSet, sid,TxBelongShard):
        newTxs = []
        newInputs: Dict[str,list] = defaultdict(list)
        testtmp = 0
        for inp in tx.TxIns:
            _,inp_sid = TxBelongShard[inp.SenptHash]
            if sid != inp_sid:
                newInputs[inp_sid].append(inp)
                #self.cst[inp_sid] += 1
            else:
                testtmp += 1

                if self.validateInput(inp, spentUTXOSet, addUTXOSet,sid) is False:
                    raise ValueError("input not valid")
        for sid,inp in newInputs.items():
            newTx = Tx("",False,tx.Size,len(inp),0,None,None,inp,None,-1,tx.TimeStamp)
            newTx.OrigTxHash = tx.Hash
            newTx.Size = self._protocol_message_size(newTx)

            newTxs.append(newTx)

        tx.OrigTxHash = tx.Hash

        tx.Hash = ""


        return newTxs

    @staticmethod
    def _protocol_message_size(tx, *, outputs=None, include_generated_hash=None):
        """Return the serialized control-message footprint in bytes.

        Cross-shard requests carry one original hash plus only the referenced
        outpoints. Responses add a generated hash and the corresponding UTXO
        proofs. They do not replicate the full original Bitcoin transaction
        once for every remote shard.
        """
        message_outputs = tx.TxOuts if outputs is None else outputs
        generated_hash = bool(tx.Hash) if include_generated_hash is None else bool(include_generated_hash)
        size = 16 + 32*bool(tx.OrigTxHash) + 32*generated_hash
        size += sum(len(inp.bytes_except_sig()) for inp in (tx.TxIns or []))
        size += sum(len(out.bytes()) for out in (message_outputs or []))
        return max(1,int(size))

    def _selected_transaction_size(self, tx, sid, placements):
        """Payload generated in this block by selecting ``tx``."""
        size = int(tx.Size)
        if placements is None:
            return size
        if not tx.OrigTxHash:
            remote_inputs = defaultdict(list)
            for inp in tx.TxIns:
                parent_sid = placements[inp.SenptHash][1]
                if parent_sid != sid:
                    remote_inputs[parent_sid].append(inp)
            for inputs in remote_inputs.values():
                request = Tx("",False,1,len(inputs),0,None,None,inputs,None,-1,tx.TimeStamp)
                request.OrigTxHash = tx.Hash
                size += self._protocol_message_size(request)
        elif not tx.Hash and tx.TxOuts is None:
            # A request is transformed into a response in this block. Reserve
            # its response bytes, including the UTXO proofs returned to origin.
            outputs = [placements[inp.SenptHash][0].TxOuts[inp.position]
                       for inp in tx.TxIns]
            size = self._protocol_message_size(
                tx,outputs=outputs,include_generated_hash=True)
        elif tx.Hash and tx.TxOuts is not None:
            # A response can complete the original and emit its final
            # transaction in the same origin-shard block. Conservatively
            # reserve both; earlier responses may therefore over-reserve but
            # can never make a valid transaction permanently ineligible.
            original = placements.get(tx.OrigTxHash)
            if original is not None:
                size += int(original[0].Size)
        return size


    def validateInput(self, iTX:TxIn, spentUTXOSet: UTXO, addUTXOSet: UTXO, sid):
        if spentUTXOSet.getOutput(iTX.SenptHash,iTX.InputAddr,iTX.position) is not None:
            raise ValueError("UTXO allready spent")

        outTx = self.utxoSets.getOutput(iTX.SenptHash,iTX.InputAddr,iTX.position)
        if outTx is None:

            outtx = addUTXOSet.getOutput(iTX.SenptHash,iTX.InputAddr,iTX.position)
            if outtx is None:
                raise ValueError("No UTXO on this input")

        return True

    def processNormalTransaction(self,tx, spentUTXO: UTXO, addUTXO: UTXO, sid):
        if self.validateNormalTransaction(tx, spentUTXO,addUTXO, sid) is False:
            return False

        for inp in tx.TxIns:
            outTx = self.utxoSets.getOutput(inp.SenptHash,inp.InputAddr,inp.position)
            if outTx is not None:
                #print(tx.Hash)
                spentUTXO.add(inp.SenptHash,outTx.Addr,outTx.position,outTx)
            else:
                outTx = addUTXO.getAndRemoveOutput(inp.SenptHash,inp.InputAddr,inp.position)
                if outTx is not None:
                    spentUTXO.add(inp.SenptHash,outTx.Addr,outTx.position,outTx)
                else:
                    raise ValueError("OutTx was not present in neither normal utxoset or addedUTXOSet")
        i = 0
        for out in tx.TxOuts:
            addUTXO.add(tx.Hash,out.Addr,i,out)
            i+=1

        return True



    def validateNormalTransaction(self, tx, spentUTXO, addUTXO, sid):
        seen = set()
        total = Decimal(0)
        for inp in tx.TxIns:
            key = (inp.SenptHash, inp.InputAddr, inp.position)
            if key in seen or spentUTXO.getOutput(*key) is not None:
                raise ValueError("UTXO already spent or duplicate input")
            seen.add(key)
            out = self.utxoSets.getOutput(*key)
            if out is None:
                out = addUTXO.getOutput(*key)
            if out is None:
                raise ValueError("No UTXO on this input")
            total += Decimal(str(out.Value))
        outputs = [Decimal(str(out.Value)) for out in tx.TxOuts]
        if any(not value.is_finite() or value < 0 for value in outputs):
            raise ValueError("Invalid output value")
        if sum(outputs, Decimal(0)) > total + Decimal('0.00000001'):
            raise ValueError("Transaction creates value")
        return True

    def whatAmI(self, tx, sid,TxBelongShard):

        if len(tx.Hash) == 64 and len(tx.OrigTxHash) == 0:
            return "normal"
        elif len(tx.Hash) == 0 and len(tx.OrigTxHash) == 64 and tx.TxOuts is None:

            return "crosstx"
        elif len(tx.Hash) == 0 and len(tx.OrigTxHash) == 64 and tx.TxOuts is not None:
            return "originaltx"
        elif len(tx.Hash) == 64 and len(tx.OrigTxHash) == 64:
            _, tx_sid = TxBelongShard[tx.OrigTxHash]

            if tx_sid != sid:

                return "crosstxresponse_C_in"
            elif tx.tx_consensus is True:
                return "crosstxresponse_C_out"
            elif sid == tx_sid:
                return "finaltransaction"


        # elif len(tx.Hash) == 64 and len(tx.OrigTxHash) == 64 and tx.tx_consensus is True:
        #     return "crosstxresponse_C_out"
        # elif len(tx.Hash) == 64 and len(tx.OrigTxHash) == 64 and sid == self.TxBelongShard[tx.OrigTxHash]:
        #     return "finaltransacion"
        else:
            return "this will never return but compiler is angry"


    def processBlock(self,txlist,sid,TxBelongShard,TxIsProcessed):
        for tx in txlist:

            types = self.whatAmI(tx,sid,TxBelongShard)

            if types == "normal" and tx.is_coinbase is True:

                TxIsProcessed[tx.Hash] = True

                i = 0
                for out in tx.TxOuts:

                    self.utxoSets.add(tx.Hash,out.Addr,i,out)
                    i += 1
            elif types == "normal" and tx.is_coinbase is False:

                TxIsProcessed[tx.Hash] = True

                tot = 0.0
                totOut = 0.0
                for inp in tx.TxIns:
                    tot = round(tot+inp.InputVout,5)
                    #print(inp.InputVout)
                    self.utxoSets.removeOutput(inp.SenptHash,inp.InputAddr,inp.position)
                i = 0
                for out in tx.TxOuts:
                    totOut = round(totOut+ out.Value,5)
                    self.utxoSets.add(tx.Hash,out.Addr,i,out)
                    i += 1

                tot = round(tot , 0)
                totOut = round(totOut,0)
                # if tot < totOut:
                #     print(tot,totOut,tx.Hash)
                #     raise ValueError('Spent value not equal to new unspent value')

            elif types == "crosstx":
                continue
            elif types == "originaltx":


                self.crossTxPool.addOriginalTx(tx)
            elif types == "crosstxresponse_C_in":

                for inp in tx.TxIns:

                    self.utxoSets.removeOutput(inp.SenptHash,inp.InputAddr,inp.position)
            elif types == "crosstxresponse_C_out":

                if tx.TxOutCount != tx.TxInCount:

                    raise ValueError('input n output length not equal idkrn')
                i = 0
                for i,out in enumerate(tx.TxOuts):

                    self.utxoSets.add(tx.TxIns[i].SenptHash,tx.TxIns[i].InputAddr,tx.TxIns[i].position,out)
                    i += 1
            elif types == "finaltransaction":

                TxIsProcessed[tx.OrigTxHash] = True

                original = self.crossTxPool.getOriginalTx(tx.OrigTxHash)
                if original is None:
                    raise ValueError('original was nil')
                expected = [(out.Addr,out.position,Decimal(str(out.Value))) for out in original.TxOuts]
                actual = [(out.Addr,out.position,Decimal(str(out.Value))) for out in tx.TxOuts]
                if actual != expected:
                    raise ValueError('Final transaction outputs differ from the admitted original')
                totOut = 0
                for inp in tx.TxIns:

                    self.utxoSets.removeOutput(inp.SenptHash,inp.InputAddr,inp.position)

                if tx.OrigTxHash != original.OrigTxHash or len(tx.OrigTxHash) == 0:

                    raise ValueError('orighasshes not equal akjb3')
                i = 0
                for out in tx.TxOuts:
                    totOut += out.Value
                    self.utxoSets.add(tx.OrigTxHash,out.Addr,i,out)
                    i = i +1
                self.crossTxPool.removeOriginalTx(tx.OrigTxHash)
            else:
                raise ValueError('unkown whatAmI')


    def create_block(self, resource, TxIsProcessed, sid, placements=None):
        # BlockSize is a number of 500-byte transaction equivalents, matching
        # the original 2000 default (approximately 1 MB); never split a tx.
        # Eq. (10) returns a transaction-equivalent service rate.  Convert it
        # to this slot's service amount, while one proposed block still cannot
        # exceed its configured payload.
        capacity = min(self.blocksize,
                       self.compute_resource(resource, types=0) * self.slot_duration)
        remaining = max(0, int(capacity * self.transaction_size))
        selected = []
        used = 0
        locally_ready = set()
        for tx in self.txPools.TxsQueue.values():
            if not tx.OrigTxHash and not all(TxIsProcessed.get(inp.SenptHash,False)
                                             or inp.SenptHash in locally_ready for inp in tx.TxIns):
                continue
            size = self._selected_transaction_size(tx,sid,placements) * p.BYTE
            if size > remaining:
                # A large candidate must not block unrelated smaller ones.
                continue
            selected.append(tx)
            remaining -= size
            used += size
            # Confirmed parents and earlier LOCAL transactions in this block
            # are usable. A cross-shard original still needs relay completion.
            if not tx.OrigTxHash and (tx.is_coinbase or
                    (placements is not None and all(placements[i.SenptHash][1] == sid for i in tx.TxIns))):
                locally_ready.add(tx.Hash)
        for tx in selected:
            self.txPools.removeTx(tx.Hash, tx.OrigTxHash)
        self.tx_list = selected
        self.last_reserved_bytes = used / p.BYTE
        cpu, bw, left = self.remove_multiple_transactions(resource, used / self.transaction_size)
        return cpu, bw, left, selected


    def remove_multiple_transactions(self, resource, total_data):
        cpu, bw = self.compute_resource(resource, total_data / self.slot_duration, types=1)
        self.cpu_needs += cpu
        self.bw_needs += bw
        return cpu, bw, [max(0, resource[0]-cpu), max(0, resource[1]-bw)]







    def compute_resource(self, resource, served=0, types=0):
        if self.resource_model == 'huang_eq2':
            # Huang et al. (TPDS 2022), Eq. (2), alpha=1/2. B_i is the
            # number of transaction equivalents dequeued in one timeslot.
            capacities = [math.sqrt(max(0, w*r)) for w, r in zip(self.w, resource)]
            capacity = sum(capacities)
            if types == 0:
                return capacity
            fraction = min(1.0, served / capacity) if capacity else 0.0
            return tuple(max(0, r) * fraction**2 for r in resource)

        if self.resource_model == 'physical_bottleneck':
            # Physical units: [CPU cores, bandwidth kb/s]. Independent
            # dimensional bottlenecks must be combined with min(), not added.
            block_capacity = self.blocksize / self.blocktime
            cpu_capacity = max(0, resource[0]) * self.cpu_tps_per_core
            bandwidth_capacity = max(0, resource[1]) * 1000 / self.transaction_size
            gross_capacity = min(block_capacity, cpu_capacity, bandwidth_capacity)
            capacity = gross_capacity * self.valid_ratio
            if types == 0:
                return capacity
            if served <= 0:
                return 0.0, 0.0
            cpu = served / (self.valid_ratio * self.cpu_tps_per_core)
            bandwidth = served * self.transaction_size / (1000 * self.valid_ratio)
            return min(max(0, resource[0]), cpu), min(max(0, resource[1]), bandwidth)

        # MSSChain Eq. (10), xi=1/2. ``blocksize`` is stored as a
        # transaction count, so S_B/S_T is already represented by it.
        # The exponent applies to each weighted resource term before the
        # terms are summed, matching the equation printed in the paper.
        coefficient = self.blocksize * self.valid_ratio / self.blocktime
        resource_factor = sum(math.sqrt(max(0, w*r)) for w, r in zip(self.w, resource))
        capacity = resource_factor * coefficient
        if types == 0:
            return capacity
        fraction = min(1.0, served / capacity) if capacity else 0.0
        return tuple(max(0, r) * fraction**2 for r in resource)








