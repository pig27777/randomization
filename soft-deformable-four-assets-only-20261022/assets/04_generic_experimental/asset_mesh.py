"""Empty two-panel bag topology, independent of the contents."""

def tabletop_mesh(nx, ny, width, height, gap, seam_width):
    """Two distinct panels joined by three side strips, with an open mouth.

    The flat panels need finite separation for thick collision shells; welding
    both panels to identical border vertices creates initial self-penetration.
    Coordinates here retain the source asset's material frame (length along Z).
    """
    import numpy as np

    points, grid, faces, regions = [], {}, [], []
    for side in (0, 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                grid[side, i, j] = len(points)
                points.append(
                    (
                        width * (i / nx - 0.5),
                        (1 if side == 0 else -1) * gap / 2,
                        0.065 + height * j / ny,
                    )
                )
        for j in range(ny):
            for i in range(nx):
                a, b, c, d = [
                    grid[side, u, v]
                    for u, v in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
                ]
                tris = (
                    ((a, b, d), (a, d, c))
                    if (i + j) % 2 == 0
                    else ((a, b, c), (b, d, c))
                )
                faces.extend(tri[::-1] if side == 0 else tri for tri in tris)
                regions.extend(
                    [int(abs(width * ((i + 0.5) / nx - 0.5)) >= width / 2 - seam_width)]
                    * 2
                )
    for i in (0, nx):
        for j in range(ny):
            a, b, c, d = [
                grid[s, i, v] for s, v in ((0, j), (1, j), (0, j + 1), (1, j + 1))
            ]
            tris = ((a, b, d), (a, d, c))
            faces.extend(tri if i == 0 else tri[::-1] for tri in tris)
            regions.extend([1, 1])
    for i in range(nx):
        a, b, c, d = [
            grid[s, u, 0] for s, u in ((0, i), (0, i + 1), (1, i), (1, i + 1))
        ]
        faces.extend(((a, b, d), (a, d, c)))
        regions.extend((1, 1))
    return (
        np.asarray(points, np.float32),
        np.asarray(faces, np.int32),
        grid,
        np.asarray(regions, np.int32),
    )
