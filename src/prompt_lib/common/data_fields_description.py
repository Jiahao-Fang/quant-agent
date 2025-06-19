## Available Markets & Symbols
DATA_FIELDS_DESCRIPTION_TEMPLATE = """
Currently available data covers:
- Time range: 2025-02-01 to 2025-02-20
- Spot market (`ST_BNS`): BTCUSDT, SOLUSDT
- Futures market (`ST_BNF`): BTCUSDT, SOLUSDT

## Available Data Sources & Fields

### BBO (`bookTicker` updates, 1-level best bid/ask snapshot)

- `exchange` (symbol): Exchange code (`ST_BNS` for spot, `ST_BNF` for futures)
- `remoteIp` / `remotePort` / `recvIp`: Source IP (ignored in factor use)
- `eventTimestamp` (int64, ns): Data event timestamp
- `transactionTimestamp` (int64, ns): Transaction time
- `recvTimestamp` (int64, ns): Received by system
- `symbol` (string): `BTCUSDT` or `SOLUSDT`
- `channel` (string): e.g. `bookTicker`
- `marketDataType`: BBO
- `lastUpdateId` (int): update version
- `htEventTimestamp`, `actionType`, `firstUpdateId`, `prevLastUpdateId`: metadata
- `bids_0_price`, `bids_0_qty`: best bid
- `asks_0_price`, `asks_0_qty`: best ask

### partial20 (`depth20` updates, top 20 level order book snapshot)

- All fields in BBO +
- `firstUpdateId`, `lastUpdateId`
- `bids_0_price` ~ `bids_19_price`, `bids_0_qty` ~ `bids_19_qty`
- `asks_0_price` ~ `asks_19_price`, `asks_0_qty` ~ `asks_19_qty`

### trade (`trade` updates)

- `tradeTimestamp` (int64, ns)
- `tradeId`, `price`, `quantity`
- `buyerOrderId`, `sellerOrderId`
- `takerSide`: BUY / SELL
- `origin`, `message_type`

---
"""