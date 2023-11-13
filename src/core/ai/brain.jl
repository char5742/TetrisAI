"頭脳"
mutable struct Brain
    main_model
    target_model
    Brain(main, target) = new(main, target)
end