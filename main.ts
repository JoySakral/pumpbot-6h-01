// pumpbot-6h-01 — momentum-breakout paper-trading bot for pump.fun bonding curves.
// Protocol: agent-fun/replay/v2, JSON lines over stdio (hello, batch, done / reply).
// Strategy (my own design): ACCUMULATE EARLY, EXIT ON A TRAILING STOP. A
// momentum signal only fires mid-spike, and the 2-slot fill latency then buys
// the top — backtests showed that costs ~35 points of return per trade. So
// instead the bot builds a position in a new mint a little at a time from its
// very first trades (at least 2 buys, buys >= 60% of volume, price not below
// its create price and never more than 30% above it), up to 0.03 SOL per
// mint. Each batch's add is capped at 1% of the mint's real SOL reserve, so
// the position only grows as other traders deepen the curve. Exits are fast:
// a stop-loss at -30% of the blended fill cost, a trailing stop 25% off the
// peak once the peak is +50% above cost (this is what captures the few
// multi-x runners that carry the strategy), or a 6000-slot time exit. One
// cycle per mint, max 5 open.
//
// Run: node main.ts

// --------------------------------------------------------------------------- types

type U64 = string;
type Pubkey = string;

interface Hello {
  type: 'hello';
  protocol: string;
  seed: U64;
  universe: { program: string; quote: string; mayhem: string; exclude_cashback: boolean; sample: { hash: string; modulus: number; residue: number } };
  capital: { lamports: U64 };
  caps: { max_open_positions: number; max_order_bps_of_real_reserve: number; max_orders_per_batch: number; shorts: boolean; leverage: boolean };
  latency_slots: number;
  fees: { protocol_bps_fallback: number; creator_bps_fallback: number; base_fee_lamports: U64; priority_fee_lamports: U64; failed_tx_bps: number };
}

interface ChainMeta {
  slot: number;
  signature: string;
  block_time: number | null;
  chain_seq: number;
}

interface TradeEvent extends ChainMeta {
  event: 'TradeEvent';
  mint: Pubkey;
  sol_amount: U64;
  token_amount: U64;
  is_buy: boolean;
  virtual_sol_reserves: U64;
  virtual_token_reserves: U64;
  real_sol_reserves: U64;
  real_token_reserves: U64;
  virtual_quote_reserves: U64;
  real_quote_reserves: U64;
  quote_amount: U64;
}

interface CreateEvent extends ChainMeta {
  event: 'CreateEvent';
  mint: Pubkey;
  virtual_token_reserves: U64;
  virtual_sol_reserves: U64;
  real_token_reserves: U64;
  virtual_quote_reserves: U64;
  is_mayhem_mode: boolean;
  is_cashback_enabled: boolean;
}

interface CompleteEvent extends ChainMeta {
  event: 'CompleteEvent';
  mint: Pubkey;
}

type ChainEvent = TradeEvent | CreateEvent | CompleteEvent;

interface Mark {
  mint: Pubkey;
  slot: number;
  virtual_token_reserves: U64;
  virtual_quote_reserves: U64;
  real_token_reserves: U64;
  real_quote_reserves: U64;
}

interface Fill {
  client_id: string;
  mint: Pubkey;
  side: 'buy' | 'sell';
  status: 'filled' | 'rejected' | 'failed';
  reason: string | null;
  decided_slot: number;
  slot: number;
  quote_lamports: U64;
  token_amount: U64;
  fee_lamports: U64;
  tx_fee_lamports: U64;
  fee_fallback: boolean;
}

interface Position {
  mint: Pubkey;
  token_amount: U64;
  cost_lamports: U64;
}

interface Batch {
  type: 'batch';
  seq: number;
  slot: number;
  events: ChainEvent[];
  marks: Mark[];
  fills: Fill[];
  account: { cash_lamports: U64; positions: Position[] };
}

interface Done {
  type: 'done';
  seq: number;
  reason: string;
  account: { cash_lamports: U64; positions: Position[] };
}

type Order =
  | { side: 'buy'; client_id: string; mint: Pubkey; quote_lamports: U64; max_slippage_bps: number }
  | { side: 'sell'; client_id: string; mint: Pubkey; token_amount: U64; max_slippage_bps: number };

// --------------------------------------------------------------------------- constants

const LAMPORTS_PER_SOL = 1_000_000_000n;
const BPS = 10_000n;

// Accumulation: build a position in a new mint a little at a time, from its very
// first trades, instead of one big buy after a momentum signal fires. Why: a
// momentum signal only fires mid-spike, and the 2-slot fill latency then buys the
// top — backtests showed that costs ~35 points of return per trade. Buying from
// the first trades gets in near the curve floor, and the 1%-of-real-reserve cap
// itself paces the build: each batch we may add at most 1% of the real reserve,
// so the position only grows as other traders' buys deepen the curve — exactly
// the selection we want. Total budget 0.03 SOL per mint.
const ACC_MAX_TRADES = 60;
const ACC_BUDGET_LAMPORTS = (LAMPORTS_PER_SOL * 3n) / 100n;
const ACC_MIN_BUY_LAMPORTS = (LAMPORTS_PER_SOL * 2n) / 1000n; // skip dust adds < 0.002 SOL
const ACC_MIN_BUY_COUNT = 2;
const ACC_MIN_BUY_SHARE_NUM = 60n;
const ACC_MIN_BUY_SHARE_DEN = 100n;
// Exits: stop loss -20%, trailing stop 15% off the peak (armed at +30% — this is
// what captures the fat right tail: a few runners pay for many small losses),
// hard time exit after 6000 slots (~40 minutes at ~0.4 s/slot). No fixed take-profit:
// backtests showed a fixed TP caps the winners that carry the whole strategy.
const STOP_LOSS_NUM = 70n;
const STOP_LOSS_DEN = 100n;
const TRAIL_ARM_NUM = 150n;
const TRAIL_ARM_DEN = 100n;
const TRAIL_DROP_NUM = 75n;
const TRAIL_DROP_DEN = 100n;
const MAX_HOLD_SLOTS = 6000;
const BUY_SLIPPAGE_BPS = 500;
const SELL_SLIPPAGE_BPS = 1000;

// --------------------------------------------------------------------------- helpers

function log(message: string): void {
  process.stderr.write(`[pumpbot] ${message}\n`);
}

/** A price as a (numerator, denominator) pair of bigints: virtual_quote / virtual_token. */
interface Price {
  n: bigint;
  d: bigint;
}

/** Compare a/b with c/d without floats: returns -1, 0, 1 for a/b ? c/d. */
function cmpRatio(aN: bigint, aD: bigint, bN: bigint, bD: bigint): number {
  const left = aN * bD;
  const right = bN * aD;
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

/** Is price p at least `num/den` times reference price r? */
function atLeastTimes(p: Price, r: Price, num: bigint, den: bigint): boolean {
  return cmpRatio(p.n, p.d, r.n * num, r.d * den) >= 0;
}

/** Is price p at most `num/den` times reference price r? */
function atMostTimes(p: Price, r: Price, num: bigint, den: bigint): boolean {
  return cmpRatio(p.n, p.d, r.n * num, r.d * den) <= 0;
}

function priceToNumber(p: Price): number {
  return Number(p.n) / Number(p.d);
}

// --------------------------------------------------------------------------- per-mint tracking

interface MintState {
  mint: Pubkey;
  createdSlot: number | null;
  createPrice: Price | null;
  price: Price | null;
  realQuote: bigint;
  trades: number;
  buyVolume: bigint;
  sellVolume: bigint;
  buyCount: number;
  complete: boolean;
  // Position bookkeeping, filled in when our buy fills land (blended across fills).
  entryPrice: Price | null;
  peakPrice: Price | null;
  entrySlot: number;
  // SOL spent and tokens received across this mint's accumulation buys.
  spent: bigint;
  accTokens: bigint;
  // A buy order is in flight for this mint; do not send a second one.
  buyPending: boolean;
}

interface PendingBuy {
  mint: Pubkey;
  price: Price;
  slot: number;
}

class Bot {
  hello: Hello | null = null;
  cash = 0n;
  positions = new Map<Pubkey, { tokens: bigint; cost: bigint }>();
  mints = new Map<Pubkey, MintState>();
  pendingBuys = new Map<string, PendingBuy>();
  tradedMints = new Set<Pubkey>();
  nextClientId = 1;
  batches = 0;
  buysSent = 0;
  sellsSent = 0;
  entriesLogged = 0;

  clientId(): string {
    const id = `b${this.nextClientId}`;
    this.nextClientId += 1;
    return id;
  }

  onHello(hello: Hello): void {
    if (hello.protocol !== 'agent-fun/replay/v2') {
      throw new Error(`unsupported protocol ${hello.protocol}`);
    }
    if (typeof hello.capital !== 'object' || hello.capital === null) {
      throw new Error('hello is missing capital');
    }
    this.hello = hello;
    this.cash = BigInt(hello.capital.lamports);
    log(`hello: capital=${hello.capital.lamports} maxPositions=${hello.caps.max_open_positions} latency=${hello.latency_slots}`);
  }

  maxBuyLamports(mint: Pubkey): bigint {
    const caps = this.hello?.caps;
    const m = this.mints.get(mint);
    if (!caps || !m || m.complete) return 0n;
    return (m.realQuote * BigInt(caps.max_order_bps_of_real_reserve)) / BPS;
  }

  committedPositions(): number {
    const set = new Set<Pubkey>(this.positions.keys());
    for (const p of this.pendingBuys.values()) set.add(p.mint);
    return set.size;
  }

  applyBatch(batch: Batch): void {
    // Fills first: confirm entries, clear pending buys, drop closed positions.
    for (const f of batch.fills) {
      if (f.side === 'buy') {
        const pending = this.pendingBuys.get(f.client_id);
        if (pending) {
          this.pendingBuys.delete(f.client_id);
          const m = this.mints.get(pending.mint);
          if (m) m.buyPending = false;
          if (f.status === 'filled' && m) {
            // True entry price, blended across accumulation fills: total SOL
            // actually spent per token received (fees included). Using the
            // decision-time curve price instead understates cost whenever the
            // price runs up during the 2-slot latency.
            const quote = BigInt(f.quote_lamports);
            const tokens = BigInt(f.token_amount);
            if (quote > 0n && tokens > 0n) {
              m.spent += quote;
              m.accTokens += tokens;
              const blended: Price = { n: m.spent, d: m.accTokens };
              m.entryPrice = blended;
              if (m.peakPrice === null || cmpRatio(blended.n, blended.d, m.peakPrice.n, m.peakPrice.d) > 0) {
                m.peakPrice = blended;
              }
              if (m.entrySlot === 0) m.entrySlot = f.slot;
            }
          }
        }
      } else if (f.side === 'sell' && f.status === 'filled') {
        this.tradedMints.add(f.mint);
      }
    }
    // Account snapshot is authoritative for cash and positions.
    this.cash = BigInt(batch.account.cash_lamports);
    this.positions.clear();
    for (const p of batch.account.positions) {
      const tokens = BigInt(p.token_amount);
      if (tokens > 0n) {
        this.positions.set(p.mint, { tokens, cost: BigInt(p.cost_lamports) });
      }
    }
    // Marks refresh reserves for mints we hold or have pending.
    for (const mark of batch.marks) {
      const m = this.mints.get(mark.mint);
      if (m === undefined) continue;
      m.price = { n: BigInt(mark.virtual_quote_reserves), d: BigInt(mark.virtual_token_reserves) };
      m.realQuote = BigInt(mark.real_quote_reserves);
    }
    // Events update every curve we track.
    for (const e of batch.events) {
      if (e.event === 'CreateEvent') {
        if (e.is_mayhem_mode || e.is_cashback_enabled) continue;
        let m = this.mints.get(e.mint);
        if (m === undefined) {
          m = {
            mint: e.mint, createdSlot: e.slot,
            createPrice: { n: BigInt(e.virtual_quote_reserves), d: BigInt(e.virtual_token_reserves) },
            price: { n: BigInt(e.virtual_quote_reserves), d: BigInt(e.virtual_token_reserves) },
            realQuote: 0n, trades: 0, buyVolume: 0n, sellVolume: 0n, buyCount: 0,
            complete: false, entryPrice: null, peakPrice: null, entrySlot: 0, spent: 0n, accTokens: 0n, buyPending: false,
          };
          this.mints.set(e.mint, m);
        }
      } else if (e.event === 'TradeEvent') {
        let m = this.mints.get(e.mint);
        if (m === undefined) {
          // First sight through a trade (create missed): seed from this trade's reserves.
          m = {
            mint: e.mint, createdSlot: null,
            createPrice: { n: BigInt(e.virtual_quote_reserves), d: BigInt(e.virtual_token_reserves) },
            price: { n: BigInt(e.virtual_quote_reserves), d: BigInt(e.virtual_token_reserves) },
            realQuote: BigInt(e.real_quote_reserves), trades: 0, buyVolume: 0n, sellVolume: 0n,
            buyCount: 0, complete: false, entryPrice: null, peakPrice: null, entrySlot: 0,
            spent: 0n, accTokens: 0n, buyPending: false,
          };
          this.mints.set(e.mint, m);
        }
        m.price = { n: BigInt(e.virtual_quote_reserves), d: BigInt(e.virtual_token_reserves) };
        m.realQuote = BigInt(e.real_quote_reserves);
        m.trades += 1;
        if (e.is_buy) {
          m.buyVolume += BigInt(e.sol_amount);
          m.buyCount += 1;
        } else {
          m.sellVolume += BigInt(e.sol_amount);
        }
        // Track the peak while holding, for the trailing stop.
        const pos = this.positions.get(e.mint);
        if (pos !== undefined && m.entryPrice !== null && m.peakPrice !== null && m.price !== null) {
          if (cmpRatio(m.price.n, m.price.d, m.peakPrice.n, m.peakPrice.d) > 0) {
            m.peakPrice = m.price;
          }
        }
      } else {
        const m = this.mints.get(e.mint);
        if (m !== undefined) m.complete = true;
      }
    }
  }

  decide(batch: Batch): Order[] {
    const orders: Order[] = [];
    const hello = this.hello;
    if (hello === null) return orders;
    const maxOrders = hello.caps.max_orders_per_batch;

    // ---- exits first: free slots and lock in gains before new entries.
    for (const [mint, pos] of this.positions) {
      if (orders.length >= maxOrders) break;
      const m = this.mints.get(mint);
      if (m === undefined || m.price === null || m.entryPrice === null) continue;
      const price = m.price;
      const entry = m.entryPrice;
      let reason: string | null = null;
      if (atMostTimes(price, entry, STOP_LOSS_NUM, STOP_LOSS_DEN)) reason = 'stop-loss';
      else if (
        m.peakPrice !== null &&
        atLeastTimes(m.peakPrice, entry, TRAIL_ARM_NUM, TRAIL_ARM_DEN) &&
        atMostTimes(price, m.peakPrice, TRAIL_DROP_NUM, TRAIL_DROP_DEN)
      ) reason = 'trailing-stop';
      else if (m.entrySlot > 0 && batch.slot - m.entrySlot > MAX_HOLD_SLOTS) reason = 'time-exit';
      if (reason !== null) {
        orders.push({
          side: 'sell', client_id: this.clientId(), mint,
          token_amount: pos.tokens.toString(), max_slippage_bps: SELL_SLIPPAGE_BPS,
        });
        this.sellsSent += 1;
        if (this.entriesLogged < 200) {
          log(`sell ${mint.slice(0, 8)} ${reason} price=${priceToNumber(price).toExponential(3)} entry=${priceToNumber(entry).toExponential(3)}`);
          this.entriesLogged += 1;
        }
        // Clear entry bookkeeping now; the fill confirms the close.
        m.entryPrice = null;
        m.peakPrice = null;
      }
    }

    // ---- accumulation: add a little to each young mint with early buy flow,
    // one buy per mint per batch, until its 0.05 SOL budget is spent. A mint we
    // already hold keeps accumulating (that is the point); a mint we already
    // sold is never re-entered.
    const seenThisBatch = new Set<Pubkey>();
    for (const e of batch.events) {
      if (orders.length >= maxOrders) break;
      if (e.event !== 'TradeEvent') continue;
      if (seenThisBatch.has(e.mint)) continue;
      seenThisBatch.add(e.mint);
      const m = this.mints.get(e.mint);
      if (m === undefined || m.price === null || m.createPrice === null) continue;
      if (m.complete || m.buyPending) continue;
      if (this.tradedMints.has(e.mint)) continue;
      if (m.trades > ACC_MAX_TRADES) continue;
      if (m.spent >= ACC_BUDGET_LAMPORTS) continue;
      // Early buy flow: at least 2 buys, buys >= 60% of volume so far, and the
      // price has not already collapsed below its create price.
      if (m.buyCount < ACC_MIN_BUY_COUNT) continue;
      const totalVolume = m.buyVolume + m.sellVolume;
      if (totalVolume === 0n) continue;
      if ((m.buyVolume * ACC_MIN_BUY_SHARE_DEN) < (totalVolume * ACC_MIN_BUY_SHARE_NUM)) continue;
      if (cmpRatio(m.price.n, m.price.d, m.createPrice.n, m.createPrice.d) < 0) continue;
      // Only accumulate while still cheap: never add once the price is 30%
      // above the create price — further adds would just raise the cost basis
      // into the top of the early spike.
      if (atLeastTimes(m.price, m.createPrice, 130n, 100n)) continue;
      // A new accumulation cycle counts against the 5-position cap; adding to
      // a mint we already hold does not.
      const isNewCycle = !this.positions.has(e.mint) && m.spent === 0n;
      if (isNewCycle && this.committedPositions() >= hello.caps.max_open_positions) continue;
      const cap = this.maxBuyLamports(e.mint);
      if (cap <= 0n) continue;
      const cap95 = (cap * 95n) / 100n;
      let size = ACC_BUDGET_LAMPORTS - m.spent;
      if (size > cap95) size = cap95;
      // Reserve cash for the tx fee and for the other open positions.
      const free = this.cash - BigInt(this.pendingBuys.size + 1) * 105000n;
      if (size > free) size = free;
      if (size < ACC_MIN_BUY_LAMPORTS) continue;
      const id = this.clientId();
      this.pendingBuys.set(id, { mint: e.mint, price: m.price, slot: batch.slot });
      m.buyPending = true;
      orders.push({
        side: 'buy', client_id: id, mint: e.mint,
        quote_lamports: size.toString(), max_slippage_bps: BUY_SLIPPAGE_BPS,
      });
      this.buysSent += 1;
      if (this.entriesLogged < 200) {
        log(`buy ${e.mint.slice(0, 8)} size=${Number(size) / 1e9} SOL spent=${Number(m.spent) / 1e9} price=${priceToNumber(m.price).toExponential(3)} trades=${m.trades} real=${Number(m.realQuote) / 1e9} SOL`);
        this.entriesLogged += 1;
      }
    }
    return orders;
  }

  onDone(done: Done): void {
    this.cash = BigInt(done.account.cash_lamports);
    log(`done (${done.reason}): batches=${this.batches} buys=${this.buysSent} sells=${this.sellsSent} cash=${this.cash} positions=${done.account.positions.length}`);
  }
}

// --------------------------------------------------------------------------- stdio loop

async function main(): Promise<void> {
  const bot = new Bot();
  let buffer = '';
  let lineNo = 0;

  process.stdin.setEncoding('utf8');
  for await (const chunk of process.stdin) {
    buffer += chunk;
    let nl = buffer.indexOf('\n');
    while (nl !== -1) {
      const line = buffer.slice(0, nl).replace(/\r$/, '');
      buffer = buffer.slice(nl + 1);
      nl = buffer.indexOf('\n');
      lineNo += 1;
      if (line.trim() === '') continue;
      let msg: unknown;
      try {
        msg = JSON.parse(line);
      } catch {
        throw new Error(`input line ${lineNo} is not JSON`);
      }
      if (typeof msg !== 'object' || msg === null || Array.isArray(msg)) {
        throw new Error(`input line ${lineNo} is not an object`);
      }
      const type = (msg as { type?: unknown }).type;
      if (type === 'hello') {
        if (bot.hello !== null) throw new Error('a second hello');
        bot.onHello(msg as Hello);
      } else if (type === 'batch') {
        const batch = msg as Batch;
        if (bot.hello === null) throw new Error('a batch before hello');
        if (!Number.isSafeInteger(batch.seq) || !Number.isSafeInteger(batch.slot) ||
            !Array.isArray(batch.events) || !Array.isArray(batch.marks) ||
            !Array.isArray(batch.fills) || typeof batch.account !== 'object' || batch.account === null) {
          throw new Error(`input line ${lineNo}: malformed batch`);
        }
        bot.applyBatch(batch);
        bot.batches += 1;
        const orders = bot.decide(batch);
        const reply = JSON.stringify({ type: 'reply', seq: batch.seq, orders });
        if (Buffer.byteLength(reply, 'utf8') > 65_536) {
          throw new Error(`batch ${batch.seq}: reply over 64 KiB`);
        }
        process.stdout.write(`${reply}\n`);
      } else if (type === 'done') {
        const done = msg as Done;
        if (bot.hello === null) throw new Error('done before hello');
        bot.onDone(done);
        return;
      } else {
        throw new Error(`input line ${lineNo}: unknown message type ${String(type)}`);
      }
    }
  }
  throw new Error('input ended before done');
}

main().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  process.stderr.write(`[pumpbot] fatal: ${message}\n`);
  process.exit(1);
});
