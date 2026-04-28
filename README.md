# OSPR Semestral Work

Simplified rotary crane model with a suspended payload on a taut rope. The crane is controlled by trolley force `F_r` and rotation torque `M_phi`.

## Files

- `symbolic_model.py` - derives the crane equations.
- `mat_model.py` - evaluates nonlinear dynamics.
- `linearization.py` - computes linearized `A, B` matrices.
- `main_lqr.py` - LQR simulation.
- `main_mpc.py` - constrained MPC simulation.
- `main_compare.py` - LQR vs MPC comparison plots and costs.

## Run

```powershell
pip install -r requirements.text
python main_compare.py
```

Run commands from the `semestral-work` directory.
