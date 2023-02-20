"探索されたゲームの状態"
struct Node
    "この状態になるための行動"
    action_list::Vector{Action}
    "操作したミノ"
    mino::Mino
    tspin::Int64
    "行動後の状態"
    game_state::GameState
end