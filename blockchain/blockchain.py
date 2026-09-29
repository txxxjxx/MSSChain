

class BlockChain(object):
    def __init__(self, ShardID, Blocks, LatestBlock, ProposedBlocks):
        self.ShardID = ShardID
        self.Blocks = Blocks
        self.LatestBlock = LatestBlock
        self.ProposedBlocks = ProposedBlocks


    def add(self, block):
        if block.ShardID != self.ShardID:
            raise ValueError('Block belongs to a different shard')
        if block.PreViousHash != self.LatestBlock:
            raise ValueError('Block does not extend the current chain tip')
        if block.contentHash() != block.Hash:
            raise ValueError('Block hash does not match its contents')
        self.Blocks.append(block)
        self.LatestBlock = block.Hash
    # def __init__(self,Hash, Height, Txs, Size, TxCnt, totalBTC,BlockReward, Parent, Next):
    #     self.Hash = Hash
    #     self.Height = Height
    #     self.Txs = Txs
    #     self.Size = Size
    #     self.TxCnt = TxCnt
    #     self.totalBTC = totalBTC
    #     self.BlockReward = BlockReward
    #     self.Parent = Parent
    #     self.Next = Next


