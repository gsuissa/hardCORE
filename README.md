# hardCORE
`hardCORE` is a package based on the tabulations of [Zeng & Sasselov (2013)](https://arxiv.org/abs/1301.0818). It can calculate the minimum core radius fraction  (minimum CRF) for a solid exoplanet using only its mass and radius (in Earth units).

If you use it, please cite it! 

## forward.nb 
`forward.nb` uses the original forward model. It calculates the planet's radius `R` (in Earth units) from its minimum CRF and mass. Only masses > 0.1 M🜨 are recommended. 

## inverter.nb
Uses Mathematica's built-in `NMinimize` to invert the forward model. You may use any minimization method you wish. `inverter.nb` accepts a file containing the planet's mass-radius joint posterior distribution in Earth units and returns the mean minimum CRF and its standard deviation. 
