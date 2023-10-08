module Reward
const reward_category = [-1, 0, 1, 2, 3, 4]

export encode_reward, decode_reward, rescaling_reward, inverse_rescaling_reward

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

"""
[-1, 0, 1, 2, 3, 4] 
報酬を6クラスに分類する
"""
function encode_reward(v)::Matrix{Float64}
    res = zeros(6, 1)
    if -1 >= v
        res[1] = 1
        return res
    elseif v > 5
        res[end] = 1
        return res
    end
    i = searchsortedfirst(reward_category, v)

    α = (v - reward_category[i-1]) / (reward_category[i] - reward_category[i-1])

    res[i] = α
    res[i-1] = 1 - α
    res
end

function decode_reward(v)
    sum(reward_category .* v)
end

end