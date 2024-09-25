from train import train
from differentials import expression, function, dx, dt

u = function()

# heat
heat = expression(
        0.1 * dx(dx(u)) == dt(u),
        x=True,
        y=True
)

heat_model, params = train(heat, epochs=20)



