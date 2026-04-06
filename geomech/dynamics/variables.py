"""System variables for Lagrangian dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SystemVariables:
    """Configuration variables that define the degrees of freedom.

    scalars:  scalar configuration variables (not yet used in EOM pipeline)
    vectors:  vector configuration variables (Vector, S2)
    matrices: matrix configuration variables (SO3)
    """

    scalars: list = field(default_factory=list)
    vectors: list = field(default_factory=list)
    matrices: list = field(default_factory=list)
