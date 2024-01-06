function rescaling_reward(v; r2d2=false)
    if r2d2
        x = abs(v) / 600
        sign(v) * ((sqrt(x + 1) - 1) + 1e-3 * x)
    else
        v / 600
    end
end

function inverse_rescaling_reward(v; r2d2=false)
    if r2d2
        ϵ = 1e-3
        x = abs(v)
        y = sign(v) * ((((sqrt(1 + 4 * ϵ * (x + 1 + ϵ)) - 1)) / (2 * ϵ))^2 - 1)
        y * 600
    else
        v * 600
    end
end
