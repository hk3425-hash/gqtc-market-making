#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>
#include "tracker.h"

namespace py = pybind11;

namespace pybind11 { namespace detail {

template <typename T>
struct type_caster<std::span<const T>> {
    PYBIND11_TYPE_CASTER(std::span<const T>, _("List[T]"));

    // Make storage a member variable tied to this specific argument's caster instance
    std::vector<T> storage;

    // Python -> C++ (Input)
    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src)) return false;
        auto seq = reinterpret_borrow<sequence>(src);

        // No more static thread_local!
        storage.reserve(seq.size());

        for (auto item : seq) {
            // Using py::cast<T> avoids GCC's dependent template syntax errors
            storage.push_back(pybind11::cast<T>(item));
        }
        value = std::span<const T>(storage);
        return true;
    }

    // C++ -> Python (Output)
    static handle cast(std::span<const T> src, return_value_policy policy, handle parent) {
        list l(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            l[i] = pybind11::cast(src[i], policy, parent);
        }
        return l.release();
    }
};

}} // namespace pybind11::detail

PYBIND11_MODULE(tracker, m) {
    m.doc() = "C++ Market Making Engine — QB / GT / BOTH modes";

    // ── TradingMode enum ──────────────────────────────────────────────────
    py::enum_<TradingMode>(m, "TradingMode")
        .value("QUEUE_BASED",  TradingMode::QUEUE_BASED)
        .value("GRID_TRADING", TradingMode::GRID_TRADING)
        .value("BOTH",         TradingMode::BOTH)
        .export_values();

    // ── Side enum ─────────────────────────────────────────────────────────
    py::enum_<Side>(m, "Side")
        .value("BUY",  Side::BUY)
        .value("SELL", Side::SELL)
        .export_values();

    // ── algo_tag constants ────────────────────────────────────────────────
    m.attr("ALGO_QB") = ALGO_QB;
    m.attr("ALGO_GT") = ALGO_GT;

    // ── ActionType constants ──────────────────────────────────────────────
    m.attr("ACTION_NONE")       = (int)ActionType::NONE;
    m.attr("ACTION_PLACE_BID")  = (int)ActionType::PLACE_BID;
    m.attr("ACTION_PLACE_ASK")  = (int)ActionType::PLACE_ASK;
    m.attr("ACTION_CANCEL_BID") = (int)ActionType::CANCEL_BID;
    m.attr("ACTION_CANCEL_ASK") = (int)ActionType::CANCEL_ASK;
    m.attr("ACTION_CANCEL_ALL") = (int)ActionType::CANCEL_ALL;

    // ── PositionState constants ───────────────────────────────────────────
    m.attr("POS_CLOSED")       = (int)PositionState::CLOSED;
    m.attr("POS_OPENING")      = (int)PositionState::OPENING;
    m.attr("POS_OPEN")         = (int)PositionState::OPEN;
    m.attr("POS_CLOSING")      = (int)PositionState::CLOSING;
    m.attr("POS_PARTIAL_FILL") = (int)PositionState::PARTIAL_FILL;
    m.attr("POS_ERROR")        = (int)PositionState::ERROR;

    // ── TradeStats ────────────────────────────────────────────────────────
    py::class_<TradeStats>(m, "TradeStats")
        .def_readonly("equity",            &TradeStats::equity)
        .def_readonly("cash",              &TradeStats::cash)
        .def_readonly("inventory",         &TradeStats::inventory)
        .def_readonly("avg_entry",         &TradeStats::avg_entry)
        .def_readonly("unrealized_pnl",    &TradeStats::unrealized_pnl)
        .def_readonly("realized_pnl",      &TradeStats::realized_pnl)
        .def_readonly("total_pnl",         &TradeStats::total_pnl)
        .def_readonly("max_drawdown",      &TradeStats::max_drawdown)
        .def_readonly("hwm",               &TradeStats::hwm)
        .def_readonly("volume",            &TradeStats::volume)
        .def_readonly("total_fees",        &TradeStats::total_fees)
        .def_readonly("qty_traded",        &TradeStats::qty_traded)
        .def_readonly("win_rate",          &TradeStats::win_rate)
        .def_readonly("fills",             &TradeStats::fills)
        .def_readonly("buy_fills",         &TradeStats::buy_fills)
        .def_readonly("sell_fills",        &TradeStats::sell_fills)
        .def_readonly("wins",              &TradeStats::wins)
        .def_readonly("losses",            &TradeStats::losses)
        .def_readonly("avg_win",           &TradeStats::avg_win)
        .def_readonly("avg_loss",          &TradeStats::avg_loss)
        .def_readonly("gross_profit",      &TradeStats::gross_profit)
        .def_readonly("gross_loss",        &TradeStats::gross_loss)
        .def_readonly("profit_factor",     &TradeStats::profit_factor)
        .def_readonly("expectancy",        &TradeStats::expectancy)
        .def_readonly("max_consec_wins",   &TradeStats::max_consec_wins)
        .def_readonly("max_consec_losses", &TradeStats::max_consec_losses)
        .def_readonly("position_state",    &TradeStats::position_state)
        .def("to_dict", [](const TradeStats& s) {
            py::dict d;
            d["equity"]            = s.equity;
            d["cash"]              = s.cash;
            d["inventory"]         = s.inventory;
            d["avg_entry"]         = s.avg_entry;
            d["unrealized_pnl"]    = s.unrealized_pnl;
            d["realized_pnl"]      = s.realized_pnl;
            d["total_pnl"]         = s.total_pnl;
            d["max_drawdown"]      = s.max_drawdown;
            d["hwm"]               = s.hwm;
            d["volume"]            = s.volume;
            d["total_fees"]        = s.total_fees;
            d["qty_traded"]        = s.qty_traded;
            d["win_rate"]          = s.win_rate;
            d["fills"]             = s.fills;
            d["buy_fills"]         = s.buy_fills;
            d["sell_fills"]        = s.sell_fills;
            d["wins"]              = s.wins;
            d["losses"]            = s.losses;
            d["avg_win"]           = s.avg_win;
            d["avg_loss"]          = s.avg_loss;
            d["gross_profit"]      = s.gross_profit;
            d["gross_loss"]        = s.gross_loss;
            d["profit_factor"]     = s.profit_factor;
            d["expectancy"]        = s.expectancy;
            d["max_consec_wins"]   = s.max_consec_wins;
            d["max_consec_losses"] = s.max_consec_losses;
            d["position_state"]    = s.position_state;
            return d;
        });

    // ── OrderAction ───────────────────────────────────────────────────────
    py::class_<OrderAction>(m, "OrderAction")
        .def_readonly("action",     &OrderAction::action)
        .def_readonly("price",      &OrderAction::price)
        .def_readonly("size",       &OrderAction::size)
        .def_readonly("grid_level", &OrderAction::grid_level)
        .def_readonly("reason",     &OrderAction::reason)
        .def_readonly("algo_tag",   &OrderAction::algo_tag)
        .def("__repr__", [](const OrderAction& a) {
            return "<OrderAction action="   + std::to_string(a.action)
                 + " algo="                + (a.algo_tag == ALGO_QB ? "QB" : "GT")
                 + " price="              + std::to_string(a.price)
                 + " size="               + std::to_string(a.size)
                 + " reason="             + a.reason + ">";
        });

    // ── TickSignal ────────────────────────────────────────────────────────
    py::class_<TickSignal>(m, "TickSignal")
        .def_readonly("mid_price",          &TickSignal::mid_price)
        .def_readonly("best_bid",           &TickSignal::best_bid)
        .def_readonly("best_ask",           &TickSignal::best_ask)
        .def_readonly("best_bid_qty",       &TickSignal::best_bid_qty)
        .def_readonly("best_ask_qty",       &TickSignal::best_ask_qty)
        .def_readonly("spread",             &TickSignal::spread)
        .def_readonly("position_state",     &TickSignal::position_state)
        .def_readonly("position_qty",       &TickSignal::position_qty)
        .def_readonly("position_value_usd", &TickSignal::position_value_usd)
        .def_readonly("equity",             &TickSignal::equity)
        .def_readonly("unrealized_pnl",     &TickSignal::unrealized_pnl)
        .def_readonly("max_drawdown",       &TickSignal::max_drawdown)
        .def("to_dict", [](const TickSignal& s) {
            py::dict d;
            d["mid_price"]          = s.mid_price;
            d["best_bid"]           = s.best_bid;
            d["best_ask"]           = s.best_ask;
            d["best_bid_qty"]       = s.best_bid_qty;
            d["best_ask_qty"]       = s.best_ask_qty;
            d["spread"]             = s.spread;
            d["position_state"]     = s.position_state;
            d["position_qty"]       = s.position_qty;
            d["position_value_usd"] = s.position_value_usd;
            d["equity"]             = s.equity;
            d["unrealized_pnl"]     = s.unrealized_pnl;
            d["max_drawdown"]       = s.max_drawdown;
            return d;
        });

    // ── FillResult ────────────────────────────────────────────────────────
    py::class_<FillResult>(m, "FillResult")
        .def_readonly("is_closing",       &FillResult::is_closing)
        .def_readonly("trade_pnl",        &FillResult::trade_pnl)
        .def_readonly("fee",              &FillResult::fee)
        .def_readonly("equity_after",     &FillResult::equity_after)
        .def_readonly("inventory_after",  &FillResult::inventory_after)
        .def_readonly("avg_entry_after",  &FillResult::avg_entry_after)
        .def_readonly("cum_realized_pnl", &FillResult::cum_realized_pnl)
        .def_readonly("cum_fees",         &FillResult::cum_fees)
        .def_readonly("cum_volume",       &FillResult::cum_volume)
        .def_readonly("cum_qty_traded",   &FillResult::cum_qty_traded)
        .def_readonly("fill_number",      &FillResult::fill_number)
        .def("to_dict", [](const FillResult& r) {
            py::dict d;
            d["is_closing"]       = r.is_closing;
            d["trade_pnl"]        = r.trade_pnl;
            d["fee"]              = r.fee;
            d["equity_after"]     = r.equity_after;
            d["inventory_after"]  = r.inventory_after;
            d["avg_entry_after"]  = r.avg_entry_after;
            d["cum_realized_pnl"] = r.cum_realized_pnl;
            d["cum_fees"]         = r.cum_fees;
            d["cum_volume"]       = r.cum_volume;
            d["cum_qty_traded"]   = r.cum_qty_traded;
            d["fill_number"]      = r.fill_number;
            return d;
        });

    // ── AsyncLogger ───────────────────────────────────────────────────────
    py::class_<AsyncLogger<>, std::shared_ptr<AsyncLogger<>>>(m, "AsyncLogger")
        .def(py::init<const std::string&>())
        .def("info",     &AsyncLogger<>::info)
        .def("warn",     &AsyncLogger<>::warn)
        .def("error",    &AsyncLogger<>::error)
        .def("decision", &AsyncLogger<>::decision);

    // ── TradeTracker ──────────────────────────────────────────────────────
    py::class_<TradeTracker<>>(m, "TradeTracker")
        .def(py::init<std::shared_ptr<AsyncLogger<>>, double>())
        .def("on_fill",            &TradeTracker<>::on_fill)
        .def("on_order_submitted", &TradeTracker<>::on_order_submitted)
        .def("on_order_cancelled", &TradeTracker<>::on_order_cancelled)
        .def("equity",             &TradeTracker<>::equity)
        .def("inventory",          &TradeTracker<>::inventory)
        .def("realized_pnl",       &TradeTracker<>::realized_pnl)
        .def("max_drawdown",       &TradeTracker<>::max_drawdown)
        .def("volume",             &TradeTracker<>::volume)
        .def("total_fees",         &TradeTracker<>::total_fees)
        .def("qty_traded",         &TradeTracker<>::qty_traded)
        .def("avg_entry",          &TradeTracker<>::avg_entry)
        .def("total_fills",        &TradeTracker<>::total_fills)
        .def("position_state",     &TradeTracker<>::position_state)
        .def("get_stats",          &TradeTracker<>::get_stats);

    // ── OrderBook ─────────────────────────────────────────────────────────
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("update",       &OrderBook::update,       py::arg("changes"))
        .def("sorted_bids",  &OrderBook::sorted_bids,  py::arg("depth") = 0)
        .def("sorted_asks",  &OrderBook::sorted_asks,  py::arg("depth") = 0)
        .def("best_bid",     &OrderBook::best_bid)
        .def("best_ask",     &OrderBook::best_ask)
        .def("bid_depth",    &OrderBook::bid_depth)
        .def("ask_depth",    &OrderBook::ask_depth)
        .def("clear",        &OrderBook::clear);

    // ── OrderBookTracker ──────────────────────────────────────────────────
    py::class_<OrderBookTracker<>>(m, "OrderBookTracker")
        .def(py::init<
            std::shared_ptr<AsyncLogger<>>,
            TradingMode,
            double,   // tick_size
            double,   // lot_size
            double,   // order_qty_usd
            // QB
            double,   // max_position_usd_qb
            double,   // qty_threshold_qb
            int,      // grid_num_qb
            // GT
            double,   // skew_gt
            double,   // half_spread_usd_gt
            double,   // grid_interval_usd_gt
            double,   // max_position_usd_gt
            int       // grid_num_gt
        >(),
            py::arg("logger"),
            py::arg("mode"),
            py::arg("tick_size"),
            py::arg("lot_size"),
            py::arg("order_qty_usd"),
            // QB
            py::arg("max_position_usd_qb"),
            py::arg("qty_threshold_qb"),
            py::arg("grid_num_qb"),
            // GT
            py::arg("skew_gt"),
            py::arg("half_spread_usd_gt"),
            py::arg("grid_interval_usd_gt"),
            py::arg("max_position_usd_gt"),
            py::arg("grid_num_gt")
        )
        .def("on_tick",
            &OrderBookTracker<>::on_tick,
            py::arg("bids"),
            py::arg("asks"),
            py::arg("inventory"),
            py::arg("active_bid_prices"),
            py::arg("active_ask_prices"),
            py::arg("active_bid_tags") = std::vector<int>{},
            py::arg("active_ask_tags") = std::vector<int>{}
        )
        .def("on_public_trade", &OrderBookTracker<>::on_public_trade)
        .def("get_ticks",       &OrderBookTracker<>::get_ticks)
        .def("get_mode",        [](const OrderBookTracker<>& t) {
            return (int)t.get_mode();
        });
}
