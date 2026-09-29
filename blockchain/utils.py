import datetime
import logging
import blockchain.params as p
from blockchain.tx import TxOut, TxIn, Tx


def data2tx(data):
    if len(data) < 8:
        logging.error("数据格式不正确")
        return None,False

    height = int(data[0])

    is_coinbase = False
    time = data[1]
    dTime = datetime.datetime.fromtimestamp(int(time), datetime.timezone.utc)
    txhash = data[2]
    if height == 91842 and txhash == "d5d27987d2a3dfc724e359870c6644b40e497bdc0589a033220fe15429d88599":
        return None,False
    if height == 91880 and txhash == "e3bf3d07d4b0375638d5f1db5255fe07ba2c4cb067cd81b84ee974b6585fb468":
        return None,False
    size = int(data[3])
    inhashs = data[5].split(";")
    inpHashs = []
    positions = []
    for inhash in inhashs:


        inphash,position = inhash.split(":")
        if inphash == "coinbase":
            break

        positions.append(int(position))
        inpHashs.append(inphash)
    inputs = []
    inutxos = data[6].split(";")
    i = 0
    for intxo in inutxos:

        inputhash, val = intxo.split(":")
        if inputhash == "coinbase":
            is_coinbase = True
        else:
            sid = tx2shard(inpHashs[i])
            inputs.append(TxIn(inpHashs[i], inputhash, positions[i],float(val),sid))

            i +=1
    outputs = []
    oututxos = data[7].split(";")

    for position, oututxo in enumerate(oututxos):
        outputhash, val = oututxo.split(":")
        sid = tx2shard(txhash)
        outputs.append(TxOut(outputhash, position,float(val),sid))
    tx = Tx(txhash, is_coinbase,size, len(inputs), len(outputs), sum(i.InputVout for i in inputs),sum(o.Value for o in outputs),inputs, outputs, height,dTime)
    return tx, True

def tx2shard(txhash):
    try:
        last16_addr = txhash[-8:]
        num = int(last16_addr, 16)
        return num % p.ShardNum
    except ValueError as err:
        logging.exception("Failed to convert address to shard: %s", err)
        raise