from enum import IntEnum

# We use the same convention as Mantaflow and FluidNet

class CellType(IntEnum):
    TypeNone = 0
    TypeFluid = 1
    TypeObstacle = 2
    TypeEmpty = 4
    TypeInflow = 8
    TypeOutflow = 16
    TypeOpen = 32
    TypeStick = 128
    TypeReserved = 256


def flagsToOccupancy(flags):
    r"""Transforms the flags tensor to occupancy tensor (0 for fluids, 1 for obstacles).

    Arguments:
        flags (Tensor): Input occupancy grid.
    Output:
        occupancy (Tensor): Output occupancy grid (0s and 1s).
    """
    occupancy = flags.clone()
    flagsFluid = occupancy.eq(CellType.TypeFluid)
    flagsObstacle = occupancy.eq(CellType.TypeObstacle)
    occupancy.masked_fill_(flagsFluid, 0)
    occupancy.masked_fill_(flagsObstacle, 1)
    return occupancy