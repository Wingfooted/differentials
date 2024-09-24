from train import train
from differentials.expression import expression, function
from differentials.differentials import dx, dt

u = function()

# heat
heat = expression(
        0.1 * dx(dx(u)) == dt(u),
        x=True,
        y=True
)

heat_model = train(heat, epochs=20)

