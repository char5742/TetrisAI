"頭脳"
mutable struct Brain
    main_model
    target_model
    mainlock::ReentrantLock
    targetlock::ReentrantLock
    Brain(main, target) = new(main, target, ReentrantLock(), ReentrantLock())
end