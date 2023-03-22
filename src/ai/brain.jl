"頭脳"
mutable struct Brain
    main_model
    target_model
    mainlock::ReentrantLock
    targetlock::ReentrantLock
    Brain(main, target) = new(main, target, ReentrantLock(), ReentrantLock())
end


"学習者"
mutable struct Learner
    brain::Brain
    "targetモデルの同期間隔"
    taget_update_cycle::Int64
    taget_update_count::Int64
    "Optimiserの状態"
    optim
end