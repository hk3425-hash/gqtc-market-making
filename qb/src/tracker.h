#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// ============================================================
// Compile-time constants
// ============================================================
inline constexpr int    MAX_OB_DEPTH         = 20;
inline constexpr int    MAX_GRID_LEVELS      = 10;
inline constexpr int    IMPULSE_WINDOW       = 600;
inline constexpr double EPSILON              = 1e-12;
inline constexpr int    LOG_BUF_SIZE         = 4096;
inline constexpr int    MAX_ACTIONS          = 512;
inline constexpr int    MAX_QUOTES_PER_SIDE  = 512;

// ============================================================
// Order-ID namespace offsets (mirrors notebook encoding)
// ============================================================
inline constexpr int ALGO_QB = 0;
inline constexpr int ALGO_GT = 1;

enum class Side : char {
    BUY  = 'b',
    SELL = 's'
};

enum class TradingMode : int {
    QUEUE_BASED  = 0,
    GRID_TRADING = 1,
    BOTH         = 2,
};

constexpr std::string_view trading_mode_str(TradingMode m) noexcept {
    switch (m) {
        case TradingMode::QUEUE_BASED:  return "QUEUE_BASED";
        case TradingMode::GRID_TRADING: return "GRID_TRADING";
        case TradingMode::BOTH:         return "BOTH";
    }
    return "UNKNOWN";
}

// ============================================================
// FixedVector — stack-allocated, zero-heap, ranges-compatible
// ============================================================
template<typename T, std::size_t Capacity>
class FixedVector {
    std::array<T, Capacity> storage_{};
    std::size_t count_{0};

public:
    using value_type      = T;
    using iterator        = typename std::array<T, Capacity>::iterator;
    using const_iterator  = typename std::array<T, Capacity>::const_iterator;

    constexpr FixedVector() noexcept(std::is_nothrow_default_constructible_v<T>) = default;

    constexpr void push_back(T const& val) {
        assert(count_ < Capacity);
        storage_[count_++] = val;
    }

    constexpr void push_back(T&& val) {
        assert(count_ < Capacity);
        storage_[count_++] = std::move(val);
    }

    template<typename... Args>
    constexpr T& emplace_back(Args&&... args) {
        assert(count_ < Capacity);
        storage_[count_] = T{std::forward<Args>(args)...};
        return storage_[count_++];
    }

    constexpr void clear() noexcept                                  { count_ = 0; }
    [[nodiscard]] constexpr bool        empty()    const noexcept    { return count_ == 0; }
    [[nodiscard]] constexpr std::size_t size()     const noexcept    { return count_; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept   { return Capacity; }

    [[nodiscard]] constexpr T&       operator[](std::size_t i)       { return storage_[i]; }
    [[nodiscard]] constexpr T const& operator[](std::size_t i) const { return storage_[i]; }

    [[nodiscard]] constexpr iterator       begin()        noexcept { return storage_.begin(); }
    [[nodiscard]] constexpr iterator       end()          noexcept { return storage_.begin() + static_cast<std::ptrdiff_t>(count_); }
    [[nodiscard]] constexpr const_iterator begin()  const noexcept { return storage_.cbegin(); }
    [[nodiscard]] constexpr const_iterator end()    const noexcept { return storage_.cbegin() + static_cast<std::ptrdiff_t>(count_); }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return storage_.cbegin(); }
    [[nodiscard]] constexpr const_iterator cend()   const noexcept { return storage_.cbegin() + static_cast<std::ptrdiff_t>(count_); }

    [[nodiscard]] constexpr std::span<T>       as_span()       noexcept { return {storage_.data(), count_}; }
    [[nodiscard]] constexpr std::span<T const> as_span() const noexcept { return {storage_.data(), count_}; }
};

// ============================================================
// Ring Buffer for Background Logging
// ============================================================
template<typename T, std::size_t Capacity>
class RingBuffer {
    std::array<T, Capacity> buffer_{};
    std::size_t head_{0};  // Write index
    std::size_t tail_{0};  // Read index
    std::size_t count_{0}; // Current size

public:
    // O(1) Circular Push: Instantly overwrites oldest data if full
    void push(T&& item) noexcept {
        buffer_[head_] = std::move(item);
        head_ = (head_ + 1) % Capacity;

        if (count_ < Capacity) {
            count_++;
        } else {
            // Overwrite oldest data: push tail forward
            tail_ = (tail_ + 1) % Capacity;
        }
    }

    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    // O(1) Pop
    bool pop(T& out) noexcept {
        if (count_ == 0) return false;
        out = std::move(buffer_[tail_]);
        tail_ = (tail_ + 1) % Capacity;
        count_--;
        return true;
    }
};

// ============================================================
// POD structs
// ============================================================
struct TradeStats {
    double equity;
    double cash;
    double inventory;
    double avg_entry;
    double unrealized_pnl;
    double realized_pnl;
    double total_pnl;
    double max_drawdown;
    double hwm;
    double volume;
    double total_fees;
    double qty_traded;
    double win_rate;
    double fills;
    double buy_fills;
    double sell_fills;
    double wins;
    double losses;
    double avg_win;
    double avg_loss;
    double gross_profit;
    double gross_loss;
    double profit_factor;
    double expectancy;
    double max_consec_wins;
    double max_consec_losses;
    double position_state;
};

enum class ActionType : int {
    NONE          = 0,
    PLACE_BID     = 1,
    PLACE_ASK     = 2,
    CANCEL_BID    = 3,
    CANCEL_ASK    = 4,
    CANCEL_ALL    = 5,
};

struct OrderAction {
    int         action;
    double      price;
    double      size;
    int         grid_level;
    std::string reason;
    int         algo_tag;
};

enum class PositionState : int {
    CLOSED       = 0,
    OPENING      = 1,
    OPEN         = 2,
    CLOSING      = 3,
    PARTIAL_FILL = 4,
    ERROR        = 5,
};

struct TickSignal {
    double mid_price{};
    double best_bid{};
    double best_ask{};
    double best_bid_qty{};
    double best_ask_qty{};
    double spread{};
    int    position_state{};
    double position_qty{};
    double position_value_usd{};
    double equity{};
    double unrealized_pnl{};
    double max_drawdown{};
};

struct GridQuote {
    double price;
    double size;
    int    level;
    int    algo_tag;
};

struct FillResult {
    bool   is_closing;
    double trade_pnl;
    double fee;
    double equity_after;
    double inventory_after;
    double avg_entry_after;
    double cum_realized_pnl;
    double cum_fees;
    double cum_volume;
    double cum_qty_traded;
    int    fill_number;
};

// ============================================================
// AsyncLogger — Heap-free struct deferred logger
// ============================================================
struct LogEvent {
    enum class Type { TEXT_INFO, TEXT_WARN, TEXT_ERR, TEXT_DECISION, TICK, FILL, STATS } type;
    std::chrono::system_clock::time_point ts;
    std::string text; // Only allocated for generic/slow text logs, never on ticks
    TradingMode mode;
    Side side;
    double d[8];
    int i[5];
};

template<std::size_t BufferReserve = LOG_BUF_SIZE>
class AsyncLogger {
private:
    std::ofstream log_file_;
    RingBuffer<LogEvent, BufferReserve> ring_queue_; // O(1) Hot-thread boundary
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_;

    void process() {
        // Zero-heap, stack-allocated batch buffer utilizing your exact reserve size
        FixedVector<LogEvent, BufferReserve> local_batch;

        while (running_.load(std::memory_order_relaxed) || !ring_queue_.empty()) {
            {
                std::unique_lock<std::mutex> lk{mtx_};
                cv_.wait(lk, [this]{ return !ring_queue_.empty() || !running_.load(); });

                // Rapidly drain the ring buffer into local_batch to free the lock instantly
                LogEvent ev;
                while (local_batch.size() < BufferReserve && ring_queue_.pop(ev)) {
                    local_batch.push_back(std::move(ev));
                }
            } // Lock released! Hot threads can continue writing.

            // Perform slow I/O string formatting outside the lock
            for (auto const& ev : local_batch) {
                auto const tt = std::chrono::system_clock::to_time_t(ev.ts);
                auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(ev.ts.time_since_epoch()) % 1000;

                std::stringstream ss;
                ss << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
                   << '.' << std::setfill('0') << std::setw(3) << ms.count();
                std::string ts_str = ss.str();

                std::stringstream out;
                if (ev.type == LogEvent::Type::TEXT_INFO)          out << ts_str << " | INFO  | " << ev.text;
                else if (ev.type == LogEvent::Type::TEXT_WARN)     out << ts_str << " | WARN  | " << ev.text;
                else if (ev.type == LogEvent::Type::TEXT_ERR)      out << ts_str << " | ERROR | " << ev.text;
                else if (ev.type == LogEvent::Type::TEXT_DECISION) out << ts_str << " | DECISION | " << ev.text;
                else if (ev.type == LogEvent::Type::TICK) {
                    out << ts_str << " | INFO  | TICK [" << trading_mode_str(ev.mode) << "]"
                        << std::fixed << " | Mid: " << std::setprecision(6) << ev.d[0]
                        << " | Spr: " << std::setprecision(8) << ev.d[1]
                        << " | OrdQty: " << std::setprecision(4) << ev.d[2]
                        << " | Inv: " << std::setprecision(4) << ev.d[3]
                        << " | OutB: " << ev.i[0] << " OutA: " << ev.i[1]
                        << " | TgtB: " << ev.i[2] << " TgtA: " << ev.i[3]
                        << " | Acts: " << ev.i[4];
                }
                else if (ev.type == LogEvent::Type::FILL) {
                    out << ts_str << " | INFO  | FILL " << (ev.side == Side::BUY ? "BUY" : "SELL") << " " << ev.d[0]
                        << std::fixed << " @ " << std::setprecision(2) << ev.d[1]
                        << " | PnL: $" << std::setprecision(2) << ev.d[2]
                        << " | Inv: "  << ev.d[3] << " | Fee: $" << ev.d[4] << " | Eq: $" << ev.d[5];
                }
                else if (ev.type == LogEvent::Type::STATS) {
                    out << ts_str << std::fixed << std::setprecision(2) << " | INFO  | STATS | Eq: $" << ev.d[0]
                        << " | RPnL: $" << ev.d[1] << " | DD: $" << ev.d[2] << " | Fees: $" << ev.d[3]
                        << " | WR: " << std::setprecision(1) << ev.d[4] << "% (" << ev.i[0] << "W/" << ev.i[1] << "L)"
                        << " | Vol: $" << std::setprecision(2) << ev.d[5] << " | Fills: " << ev.i[2]
                        << " (" << ev.i[3] << "B/" << ev.i[4] << "S)";
                }

                std::string final_msg = out.str();
                if (log_file_.is_open()) log_file_ << final_msg << '\n';
                std::cout << final_msg << '\n';
            }

            if (!local_batch.empty() && log_file_.is_open()) log_file_.flush();
            local_batch.clear(); // O(1) clear, memory is reused next cycle
        }
    }

    void enqueue_event(LogEvent&& ev) {
        {
            std::lock_guard<std::mutex> lk{mtx_};
            ring_queue_.push(std::move(ev));
        }
        cv_.notify_one();
    }

public:
    explicit AsyncLogger(std::string const& tag) : running_{true} {
        std::filesystem::create_directories("logs");
        auto const epoch = std::chrono::seconds(std::time(nullptr)).count();
        std::string const path{"logs/" + tag + "_" + std::to_string(epoch) + ".log"};
        log_file_.open(path, std::ios::out | std::ios::app);

        // No more string vectors to reserve. Everything is safely pre-allocated in arrays.
        worker_ = std::thread(&AsyncLogger::process, this);
    }

    ~AsyncLogger() {
        running_.store(false);
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
        if (log_file_.is_open()) log_file_.close();
    }

    AsyncLogger(AsyncLogger const&)            = delete;
    AsyncLogger& operator=(AsyncLogger const&) = delete;

    void info(std::string const& msg) {
        LogEvent ev{}; ev.type = LogEvent::Type::TEXT_INFO; ev.ts = std::chrono::system_clock::now(); ev.text = msg;
        enqueue_event(std::move(ev));
    }
    void warn(std::string const& msg) {
        LogEvent ev{}; ev.type = LogEvent::Type::TEXT_WARN; ev.ts = std::chrono::system_clock::now(); ev.text = msg;
        enqueue_event(std::move(ev));
    }
    void error(std::string const& msg) {
        LogEvent ev{}; ev.type = LogEvent::Type::TEXT_ERR; ev.ts = std::chrono::system_clock::now(); ev.text = msg;
        enqueue_event(std::move(ev));
    }
    void decision(std::string const& msg) {
        LogEvent ev{}; ev.type = LogEvent::Type::TEXT_DECISION; ev.ts = std::chrono::system_clock::now(); ev.text = msg;
        enqueue_event(std::move(ev));
    }

    // Zero-string-allocation logging methods for hot threads
    void log_tick(TradingMode mode, double mid, double spread, double order_qty, double inventory,
                  int ob, int oa, int tgt_b, int tgt_a, int acts) {
        LogEvent ev{};
        ev.type = LogEvent::Type::TICK; ev.ts = std::chrono::system_clock::now(); ev.mode = mode;
        ev.d[0] = mid; ev.d[1] = spread; ev.d[2] = order_qty; ev.d[3] = inventory;
        ev.i[0] = ob; ev.i[1] = oa; ev.i[2] = tgt_b; ev.i[3] = tgt_a; ev.i[4] = acts;
        enqueue_event(std::move(ev));
    }

    void log_fill(Side side, double size, double price, double trade_pnl, double inventory, double fee, double eq) {
        LogEvent ev{};
        ev.type = LogEvent::Type::FILL; ev.ts = std::chrono::system_clock::now(); ev.side = side;
        ev.d[0] = size; ev.d[1] = price; ev.d[2] = trade_pnl; ev.d[3] = inventory; ev.d[4] = fee; ev.d[5] = eq;
        enqueue_event(std::move(ev));
    }

    void log_stats(double eq, double realized_pnl, double max_dd, double total_fees, double wr,
                   int wins, int losses, double total_volume, int fills, int buy_fills, int sell_fills) {
        LogEvent ev{};
        ev.type = LogEvent::Type::STATS; ev.ts = std::chrono::system_clock::now();
        ev.d[0] = eq; ev.d[1] = realized_pnl; ev.d[2] = max_dd; ev.d[3] = total_fees; ev.d[4] = wr; ev.d[5] = total_volume;
        ev.i[0] = wins; ev.i[1] = losses; ev.i[2] = fills; ev.i[3] = buy_fills; ev.i[4] = sell_fills;
        enqueue_event(std::move(ev));
    }
};

// ============================================================
// TradeTracker (PnL, position state machine, risk metrics)
//   LogBufSize: forwarded to AsyncLogger template
// ============================================================
template<std::size_t LogBufSize = LOG_BUF_SIZE>
class TradeTracker {
private:
    std::shared_ptr<AsyncLogger<LogBufSize>> log_;
    double cash_{0.0};
    double inventory_{0.0};
    double avg_entry_{0.0};
    double total_volume_{0.0};
    double fee_rate_;

    int    fills_{0};
    int    buy_fills_{0};
    int    sell_fills_{0};
    int    wins_{0};
    int    losses_{0};
    int    consecutive_wins_{0};
    int    consecutive_losses_{0};
    int    max_consecutive_wins_{0};
    int    max_consecutive_losses_{0};
    double hwm_{0.0};
    double max_dd_{0.0};
    double realized_pnl_{0.0};
    double total_fees_{0.0};
    double total_qty_traded_{0.0};
    double sum_win_pnl_{0.0};
    double sum_loss_pnl_{0.0};
    double gross_profit_{0.0};
    double gross_loss_{0.0};

    PositionState pos_state_{PositionState::CLOSED};
    bool   has_pending_buy_{false};
    bool   has_pending_sell_{false};
    double pending_buy_qty_{0.0};
    double pending_sell_qty_{0.0};

    std::mutex mtx_;

public:
    TradeTracker(std::shared_ptr<AsyncLogger<LogBufSize>> log, double fee_rate)
        : log_{std::move(log)}, fee_rate_{fee_rate} {}

    FillResult on_fill(Side side, double price, double size) {
        std::lock_guard<std::mutex> lk{mtx_};

        assert(price > 0 && size > 0);
        fills_++;
        if (side == Side::BUY) buy_fills_++; else sell_fills_++;

        double const notional{price * size};
        double const fee{notional * fee_rate_};
        total_volume_     += notional;
        total_fees_       += fee;
        total_qty_traded_ += size;

        bool const closing{(side == Side::SELL && inventory_ > EPSILON)
                        || (side == Side::BUY  && inventory_ < -EPSILON)};
        double trade_pnl{0.0};
        if (closing && avg_entry_ > 0) {
            double const raw{(side == Side::SELL)
                           ? (price - avg_entry_) * size
                           : (avg_entry_ - price) * size};
            trade_pnl = raw - fee - (avg_entry_ * size * fee_rate_);
            realized_pnl_ += trade_pnl;

            if (trade_pnl > 0) {
                wins_++;
                sum_win_pnl_ += trade_pnl;
                gross_profit_ += trade_pnl;
                consecutive_wins_++;
                consecutive_losses_ = 0;
                if (consecutive_wins_ > max_consecutive_wins_)
                    max_consecutive_wins_ = consecutive_wins_;
            } else {
                losses_++;
                sum_loss_pnl_ += trade_pnl;
                gross_loss_ += std::abs(trade_pnl);
                consecutive_losses_++;
                consecutive_wins_ = 0;
                if (consecutive_losses_ > max_consecutive_losses_)
                    max_consecutive_losses_ = consecutive_losses_;
            }
        }

        if (side == Side::BUY) {
            if (inventory_ >= 0)
                avg_entry_ = (avg_entry_ * inventory_ + notional) / (inventory_ + size);
            cash_      -= (notional + fee);
            inventory_ += size;
        } else {
            if (inventory_ <= 0)
                avg_entry_ = (avg_entry_ * std::abs(inventory_) + notional)
                           / (std::abs(inventory_) + size);
            cash_      += (notional - fee);
            inventory_ -= size;
        }
        if (std::abs(inventory_) < EPSILON) {
            avg_entry_ = 0.0;
            inventory_ = 0.0;
        }

        double const eq{cash_ + inventory_ * price};
        if (eq > hwm_) hwm_ = eq;
        if (hwm_ - eq > max_dd_) max_dd_ = hwm_ - eq;

        update_position_state();

        if (fills_ % 5 == 0) {
            double const wr{(wins_ + losses_ > 0)
                          ? 100.0 * wins_ / (wins_ + losses_) : 0.0};
            std::stringstream ss;
            ss << std::fixed << std::setprecision(8);
            ss << "STATS | Eq: $" << eq
               << " | RPnL: $" << realized_pnl_
               << " | DD: $"   << max_dd_
               << " | Fees: $" << total_fees_
               << " | WR: "    << std::setprecision(3) << wr
               << "% (" << wins_ << "W/" << losses_ << "L)"
               << " | Vol: $"  << std::setprecision(8) << total_volume_
               << " | Fills: " << fills_
               << " (" << buy_fills_ << "B/" << sell_fills_ << "S)";
            log_->info(ss.str());
        } else {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(8);
            ss << "FILL " << (side == Side::BUY ? "BUY" : "SELL") << " " << size
               << " @ " << std::setprecision(3) << price
               << " | PnL: $" << std::setprecision(8) << trade_pnl
               << " | Inv: "  << inventory_
               << " | Fee: $" << fee
               << " | Eq: $"  << eq;
            log_->info(ss.str());
        }

        return FillResult{
            closing, trade_pnl, fee, eq, inventory_, avg_entry_,
            realized_pnl_, total_fees_, total_volume_, total_qty_traded_, fills_
        };
    }

    void on_order_submitted(Side side, double qty) {
        std::lock_guard<std::mutex> lk{mtx_};
        if (side == Side::BUY) { has_pending_buy_ = true;  pending_buy_qty_  = qty; }
        else                   { has_pending_sell_ = true; pending_sell_qty_ = qty; }
        update_position_state();
    }

    void on_order_cancelled(Side side) {
        std::lock_guard<std::mutex> lk{mtx_};
        if (side == Side::BUY) { has_pending_buy_ = false;  pending_buy_qty_  = 0.0; }
        else                   { has_pending_sell_ = false; pending_sell_qty_ = 0.0; }
        update_position_state();
    }

    [[nodiscard]] double equity(double mid)  const noexcept { return cash_ + inventory_ * mid; }
    [[nodiscard]] double inventory()         const noexcept { return inventory_; }
    [[nodiscard]] double realized_pnl()      const noexcept { return realized_pnl_; }
    [[nodiscard]] double max_drawdown()      const noexcept { return max_dd_; }
    [[nodiscard]] double volume()            const noexcept { return total_volume_; }
    [[nodiscard]] double total_fees()        const noexcept { return total_fees_; }
    [[nodiscard]] double qty_traded()        const noexcept { return total_qty_traded_; }
    [[nodiscard]] double avg_entry()         const noexcept { return avg_entry_; }
    [[nodiscard]] int    total_fills()       const noexcept { return fills_; }
    [[nodiscard]] int    position_state()    const noexcept { return static_cast<int>(pos_state_); }

    [[nodiscard]] TradeStats get_stats(double mid) {
        std::lock_guard<std::mutex> lk{mtx_};
        double const eq{cash_ + inventory_ * mid};
        double const unr{(avg_entry_ > 0 && std::abs(inventory_) > EPSILON)
                       ? (mid - avg_entry_) * inventory_ : 0.0};
        double const wr{(wins_+losses_>0) ? 100.0*wins_/(wins_+losses_) : 0.0};
        double const avg_win{(wins_   > 0) ? sum_win_pnl_  / wins_   : 0.0};
        double const avg_loss{(losses_ > 0) ? sum_loss_pnl_ / losses_ : 0.0};
        double const pf{(gross_loss_ > EPSILON) ? gross_profit_ / gross_loss_ : 0.0};
        double const exp{(wins_+losses_>0)
            ? (wr/100.0 * avg_win) + ((1.0 - wr/100.0) * avg_loss) : 0.0};

        return TradeStats{
            eq, cash_, inventory_, avg_entry_, unr, realized_pnl_, realized_pnl_ + unr,
            max_dd_, hwm_, total_volume_, total_fees_, total_qty_traded_, wr,
            static_cast<double>(fills_), static_cast<double>(buy_fills_),
            static_cast<double>(sell_fills_), static_cast<double>(wins_),
            static_cast<double>(losses_), avg_win, avg_loss, gross_profit_, gross_loss_,
            pf, exp, static_cast<double>(max_consecutive_wins_),
            static_cast<double>(max_consecutive_losses_),
            static_cast<double>(static_cast<int>(pos_state_))
        };
    }

private:
    void update_position_state() noexcept {
        bool const has_holdings{std::abs(inventory_) > EPSILON};
        bool const has_buy_ord{has_pending_buy_};
        bool const has_sell_ord{has_pending_sell_};

        if (!has_buy_ord && !has_sell_ord && !has_holdings)
            pos_state_ = PositionState::CLOSED;
        else if (!has_buy_ord && !has_sell_ord && has_holdings)
            pos_state_ = PositionState::OPEN;
        else if (has_buy_ord && !has_sell_ord && !has_holdings)
            pos_state_ = PositionState::OPENING;
        else if (has_buy_ord && !has_sell_ord && has_holdings)
            pos_state_ = PositionState::PARTIAL_FILL;
        else if (!has_buy_ord && has_sell_ord && has_holdings)
            pos_state_ = PositionState::CLOSING;
        else if (has_buy_ord && has_sell_ord && has_holdings)
            pos_state_ = PositionState::OPEN;
        else {
            pos_state_ = PositionState::ERROR;
            log_->error("Position state ERROR: buy_ord=" + std::to_string(has_buy_ord)
                      + " sell_ord=" + std::to_string(has_sell_ord)
                      + " holdings=" + std::to_string(has_holdings));
        }
    }
};

// ============================================================
// OrderBook — sorted order book with O(log n) updates
// ============================================================
class OrderBook {
private:
    std::map<double, double, std::greater<>> bids_;
    std::map<double, double>                 asks_;

public:
    OrderBook() = default;

    void update(std::span<std::tuple<std::string, double, double> const> changes) {
        for (auto const& [side, px, sz] : changes) {
            if (side == "buy") {
                if (sz <= 0.0) bids_.erase(px);
                else           bids_[px] = sz;
            } else {
                if (sz <= 0.0) asks_.erase(px);
                else           asks_[px] = sz;
            }
        }
    }

    [[nodiscard]] std::vector<std::pair<double,double>> sorted_bids(int depth = 0) const {
        int const limit{(depth > 0) ? depth : static_cast<int>(bids_.size())};
        std::vector<std::pair<double,double>> out;
        out.reserve(static_cast<std::size_t>(std::min(limit, static_cast<int>(bids_.size()))));
        for (auto const& [px, sz] : bids_ | std::views::take(limit))
            out.emplace_back(px, sz);
        return out;
    }

    [[nodiscard]] std::vector<std::pair<double,double>> sorted_asks(int depth = 0) const {
        int const limit{(depth > 0) ? depth : static_cast<int>(asks_.size())};
        std::vector<std::pair<double,double>> out;
        out.reserve(static_cast<std::size_t>(std::min(limit, static_cast<int>(asks_.size()))));
        for (auto const& [px, sz] : asks_ | std::views::take(limit))
            out.emplace_back(px, sz);
        return out;
    }

    [[nodiscard]] double best_bid()  const noexcept { return bids_.empty() ? 0.0 : bids_.begin()->first; }
    [[nodiscard]] double best_ask()  const noexcept { return asks_.empty() ? 0.0 : asks_.begin()->first; }
    [[nodiscard]] int    bid_depth() const noexcept { return static_cast<int>(bids_.size()); }
    [[nodiscard]] int    ask_depth() const noexcept { return static_cast<int>(asks_.size()); }
    void clear() noexcept { bids_.clear(); asks_.clear(); }
};

// ============================================================
// OrderBookTracker — unified quoting engine
//
// Template parameters:
//   MaxQuotesPerSide — FixedVector capacity for target bids/asks
//   MaxActions       — FixedVector capacity for actions per tick
//   LogBufSize       — forwarded to AsyncLogger
//
// Hot-path (on_tick) is zero-heap: all scratch storage lives in
// stack-allocated FixedVectors that are .clear()'d each tick.
// ============================================================
template<std::size_t MaxQuotesPerSide = MAX_QUOTES_PER_SIDE,
         std::size_t MaxActions       = MAX_ACTIONS,
         std::size_t LogBufSize       = LOG_BUF_SIZE>
class OrderBookTracker {
private:
    std::shared_ptr<AsyncLogger<LogBufSize>> log_;
    TradingMode mode_;

    // ── Shared ──────────────────────────────────────────────
    double tick_size_;
    double lot_size_;
    double order_qty_usd_;

    // ── QB parameters ───────────────────────────────────────
    double max_position_usd_qb_;
    double qty_threshold_qb_;
    int    grid_num_qb_;

    // ── GT parameters ───────────────────────────────────────
    double skew_gt_;
    double half_spread_usd_gt_;
    double grid_interval_usd_gt_;
    double max_position_usd_gt_;
    int    grid_num_gt_;
    double requote_tol_gt_;

    int    tick_count_{0};

    // ── Pre-allocated scratch buffers (zero heap on hot path) ─
    mutable FixedVector<GridQuote,   MaxQuotesPerSide> target_bids_;
    mutable FixedVector<GridQuote,   MaxQuotesPerSide> target_asks_;
    mutable FixedVector<OrderAction, MaxActions>       actions_;

    // ── Internal helpers ────────────────────────────────────

    [[nodiscard]] double compute_order_qty(double mid) const noexcept {
        double const raw{order_qty_usd_ / mid};
        return std::max(std::round(raw / lot_size_) * lot_size_, lot_size_);
    }

    // ── QB sub-engine ────────────────────────────────────────
    void compute_qb_quotes(
        double best_bid, double best_ask, double best_bid_qty, double best_ask_qty,
        double inventory, double order_qty,
        FixedVector<GridQuote, MaxQuotesPerSide>& target_bids,
        FixedVector<GridQuote, MaxQuotesPerSide>& target_asks) const noexcept
    {
        double const mid{(best_bid + best_ask) / 2.0};
        double const position_value_usd{inventory * mid};
        double const skew_val{(order_qty > 0) ? inventory / order_qty : 0.0};

        double bid_price{(best_bid_qty < qty_threshold_qb_ && skew_val > 0)
                       ? best_bid - tick_size_ : best_bid};
        double ask_price{(best_ask_qty < qty_threshold_qb_ && skew_val < 0)
                       ? best_ask + tick_size_ : best_ask};

        bid_price = std::floor(bid_price / tick_size_) * tick_size_;
        ask_price = std::ceil( ask_price / tick_size_) * tick_size_;

        if (position_value_usd < max_position_usd_qb_ && std::isfinite(bid_price)) {
            double const remaining{max_position_usd_qb_ - position_value_usd};
            int const max_safe{std::max(0, std::min(grid_num_qb_,
                               static_cast<int>(std::floor((remaining / mid) / order_qty))))};
            double p{bid_price};
            for (int i{0}; i < max_safe; ++i) {
                target_bids.push_back({p, order_qty, i, ALGO_QB});
                p -= tick_size_;
            }
        }

        if (inventory >= order_qty && std::isfinite(ask_price)) {
            int const safe_levels{std::min(grid_num_qb_,
                                  static_cast<int>(std::floor(inventory / order_qty)))};
            double p{ask_price};
            for (int i{0}; i < safe_levels; ++i) {
                target_asks.push_back({p, order_qty, i, ALGO_QB});
                p += tick_size_;
            }
        }
    }

    // ── GT sub-engine ────────────────────────────────────────
    void compute_gt_quotes(
        double best_bid, double best_ask,
        double inventory, double order_qty,
        FixedVector<GridQuote, MaxQuotesPerSide>& target_bids,
        FixedVector<GridQuote, MaxQuotesPerSide>& target_asks) const noexcept
    {
        double const mid{(best_bid + best_ask) / 2.0};
        double const position_value_usd{inventory * mid};

        double const reservation{mid - skew_gt_ * inventory};
        double bid_price{std::min(reservation - half_spread_usd_gt_, best_bid)};
        double ask_price{std::max(reservation + half_spread_usd_gt_, best_ask)};

        bid_price = std::floor(bid_price / grid_interval_usd_gt_) * grid_interval_usd_gt_;
        ask_price = std::ceil( ask_price / grid_interval_usd_gt_) * grid_interval_usd_gt_;
        //bid_price = std::floor(bid_price / tick_size_) * tick_size_;
        //ask_price = std::ceil( ask_price / tick_size_) * tick_size_;

        if (position_value_usd < max_position_usd_gt_ && std::isfinite(bid_price)) {
            double const remaining{max_position_usd_gt_ - position_value_usd};
            int const max_safe{std::max(0, std::min(grid_num_gt_,
                               static_cast<int>(std::floor(remaining / order_qty_usd_))))};
            double p{bid_price};
            for (int i{0}; i < max_safe; ++i) {
                target_bids.push_back({p, order_qty, i, ALGO_GT});
                p -= grid_interval_usd_gt_;
            }
        }

        if (inventory >= order_qty && std::isfinite(ask_price)) {
            int const safe_levels{std::min(grid_num_gt_,
                                  static_cast<int>(std::floor(inventory / order_qty)))};
            double p{ask_price};
            for (int i{0}; i < safe_levels; ++i) {
                target_asks.push_back({p, order_qty, i, ALGO_GT});
                p += grid_interval_usd_gt_;
            }
        }
    }

    // ── Reconcile: cancel stale, place new ───────────────────
    void reconcile(
        std::span<GridQuote const>             targets,
        std::span<std::optional<double> const> active_prices,
        std::span<int const>                   active_tags,
        double     tol,
        int        algo_tag,
        ActionType place_action,
        ActionType cancel_action,
        FixedVector<OrderAction, MaxActions>&   actions) const
    {
        // Cancel stale active orders belonging to this algo
        for (int i{0}; i < static_cast<int>(active_prices.size()); ++i) {
            if (!active_prices[i].has_value()) continue;
            if (active_tags[i] != algo_tag) continue;
            double const ap{active_prices[i].value()};

            bool const wanted{std::ranges::any_of(targets, [&](auto const& tq) {
                return tq.algo_tag == algo_tag && std::abs(tq.price - ap) <= tol;
            })};

            if (!wanted)
                actions.push_back({static_cast<int>(cancel_action), ap, 0.0, i,
                    (algo_tag == ALGO_QB ? "stale_qb" : "stale_gt"), algo_tag});
        }

        // Place new target orders that aren't yet active
        for (auto const& tq : targets | std::views::filter(
                 [algo_tag](auto const& q){ return q.algo_tag == algo_tag; }))
        {
            bool const already{std::ranges::any_of(
                std::views::iota(0, static_cast<int>(active_prices.size())),
                [&](int i) {
                    return active_prices[i].has_value()
                        && active_tags[i] == algo_tag
                        && std::abs(tq.price - active_prices[i].value()) <= tol;
                })};

            if (!already)
                actions.push_back({static_cast<int>(place_action), tq.price, tq.size, tq.level,
                    (algo_tag == ALGO_QB ? "new_qb" : "new_gt"), algo_tag});
        }
    }

public:
    // ── Constructor ─────────────────────────────────────────
    OrderBookTracker(
        std::shared_ptr<AsyncLogger<LogBufSize>> log,
        TradingMode mode,
        double tick_size,
        double lot_size,
        double order_qty_usd,
        // QB
        double max_position_usd_qb,
        double qty_threshold_qb,
        int    grid_num_qb,
        // GT
        double skew_gt,
        double half_spread_usd_gt,
        double grid_interval_usd_gt,
        double max_position_usd_gt,
        int    grid_num_gt
    )
        : log_{std::move(log)}, mode_{mode},
          tick_size_{tick_size}, lot_size_{lot_size},
          order_qty_usd_{order_qty_usd},
          max_position_usd_qb_{max_position_usd_qb},
          qty_threshold_qb_{qty_threshold_qb},
          grid_num_qb_{grid_num_qb},
          skew_gt_{skew_gt},
          half_spread_usd_gt_{half_spread_usd_gt},
          grid_interval_usd_gt_{grid_interval_usd_gt},
          max_position_usd_gt_{max_position_usd_gt},
          grid_num_gt_{grid_num_gt},
          requote_tol_gt_{grid_interval_usd_gt}
    {
        log_->info("OrderBookTracker mode=" + std::string{trading_mode_str(mode_)});
    }

    void on_public_trade(std::string_view /*side*/, double /*size*/) noexcept {}

    // ── CORE TICK ────────────────────────────────────────────
    // Returns TickSignal + span into the internal actions_ buffer.
    // The span is valid until the next on_tick call.
    std::pair<TickSignal, std::span<OrderAction const>> on_tick(
        std::span<std::pair<double,double> const> bids,
        std::span<std::pair<double,double> const> asks,
        double inventory,
        std::span<std::optional<double> const> active_bid_prices,
        std::span<std::optional<double> const> active_ask_prices,
        std::span<int const> active_bid_tags = {},
        std::span<int const> active_ask_tags = {}
    ) {
        TickSignal sig{};

        // Clear scratch buffers — no alloc, stack memory reused
        target_bids_.clear();
        target_asks_.clear();
        actions_.clear();

        if (bids.empty() || asks.empty()) return {sig, actions_.as_span()};
        tick_count_++;

        auto const& [best_bid, best_bid_qty] = bids[0];
        auto const& [best_ask, best_ask_qty] = asks[0];
        double const mid{(best_bid + best_ask) / 2.0};
        double const order_qty{compute_order_qty(mid)};

        // Default tags to ALGO_GT if not provided (backward-compat)
        FixedVector<int, MaxQuotesPerSide> synth_bid_tags;
        FixedVector<int, MaxQuotesPerSide> synth_ask_tags;

        std::span<int const> bid_tags{active_bid_tags};
        std::span<int const> ask_tags{active_ask_tags};

        if (active_bid_tags.empty() && !active_bid_prices.empty()) {
            for (std::size_t i{0}; i < active_bid_prices.size(); ++i)
                synth_bid_tags.push_back(ALGO_GT);
            bid_tags = synth_bid_tags.as_span();
        }
        if (active_ask_tags.empty() && !active_ask_prices.empty()) {
            for (std::size_t i{0}; i < active_ask_prices.size(); ++i)
                synth_ask_tags.push_back(ALGO_GT);
            ask_tags = synth_ask_tags.as_span();
        }

        // ── Dispatch to sub-engines based on mode ────────────
        if (mode_ == TradingMode::QUEUE_BASED || mode_ == TradingMode::BOTH) {
            compute_qb_quotes(best_bid, best_ask, best_bid_qty, best_ask_qty,
                              inventory, order_qty,
                              target_bids_, target_asks_);
        }
        if (mode_ == TradingMode::GRID_TRADING || mode_ == TradingMode::BOTH) {
            compute_gt_quotes(best_bid, best_ask, inventory, order_qty,
                              target_bids_, target_asks_);
        }

        auto const tgt_bids_span = target_bids_.as_span();
        auto const tgt_asks_span = target_asks_.as_span();

        // ── Reconcile bids ───────────────────────────────────
        if (mode_ == TradingMode::QUEUE_BASED || mode_ == TradingMode::BOTH) {
            reconcile(tgt_bids_span, active_bid_prices, bid_tags,
                      tick_size_, ALGO_QB,
                      ActionType::PLACE_BID, ActionType::CANCEL_BID, actions_);
        }
        if (mode_ == TradingMode::GRID_TRADING || mode_ == TradingMode::BOTH) {
            reconcile(tgt_bids_span, active_bid_prices, bid_tags,
                      requote_tol_gt_, ALGO_GT,
                      ActionType::PLACE_BID, ActionType::CANCEL_BID, actions_);
        }

        // ── Reconcile asks ───────────────────────────────────
        if (mode_ == TradingMode::QUEUE_BASED || mode_ == TradingMode::BOTH) {
            reconcile(tgt_asks_span, active_ask_prices, ask_tags,
                      tick_size_, ALGO_QB,
                      ActionType::PLACE_ASK, ActionType::CANCEL_ASK, actions_);
        }
        if (mode_ == TradingMode::GRID_TRADING || mode_ == TradingMode::BOTH) {
            reconcile(tgt_asks_span, active_ask_prices, ask_tags,
                      requote_tol_gt_, ALGO_GT,
                      ActionType::PLACE_ASK, ActionType::CANCEL_ASK, actions_);
        }

        // ── Throttled logging ────────────────────────────────
        if (tick_count_ % 100 == 0) {
            auto const ob = std::ranges::count_if(active_bid_prices,
                                [](auto const& p){ return p.has_value(); });
            auto const oa = std::ranges::count_if(active_ask_prices,
                                [](auto const& p){ return p.has_value(); });
            std::stringstream ss;
            ss << std::fixed;
            ss << "TICK [" << trading_mode_str(mode_) << "]"
               << " | Mid: " << std::setprecision(6) << mid
               << " | Spr: " << std::setprecision(8) << (best_ask - best_bid)
               << " | OrdQty: " << std::setprecision(4) << order_qty
               << " | Inv: " << inventory
               << " | OutB: " << ob << " OutA: " << oa
               << " | TgtB: " << target_bids_.size()
               << " TgtA: " << target_asks_.size()
               << " | Acts: " << actions_.size();
            log_->info(ss.str());
        }

        // ── Populate signal ──────────────────────────────────
        double const pos_usd{inventory * mid};
        double const spread{best_ask - best_bid};

        sig.mid_price          = mid;
        sig.best_bid           = best_bid;
        sig.best_ask           = best_ask;
        sig.best_bid_qty       = best_bid_qty;
        sig.best_ask_qty       = best_ask_qty;
        sig.spread             = spread;
        sig.position_qty       = inventory;
        sig.position_value_usd = pos_usd;

        return {sig, actions_.as_span()};
    }

    [[nodiscard]] int         get_ticks() const noexcept { return tick_count_; }
    [[nodiscard]] TradingMode get_mode()  const noexcept { return mode_; }
};
