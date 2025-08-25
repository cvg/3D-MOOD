# Coordinate System

We use different 3D bounding box coordinate system compared to Omni3D.

But to use the PyTorch 3D for evaluation we will convert the final output to the same coordinate as Omni3D.

## Omni3D

All 3D annotations are provided in a shared camera coordinate system with +x right, +y down, +z toward screen. 

The vertex order of bbox3D_cam:
```python
                v4_____________________v5
                /|                    /|
               / |                   / |
              /  |                  /  |
             /___|_________________/   |
          v0|    |                 |v1 |
            |    |                 |   |
            |    |                 |   |
            |    |                 |   |
            |    |_________________|___|
            |   / v7               |   /v6
            |  /                   |  /
            | /                    | /
            |/_____________________|/
            v3                     v2
```

## Ours

```python
               (back)
        (6) +---------+. (7)
            | ` .     |  ` .
            | (4) +---+-----+ (5)
            |     |   |     |
        (2) +-----+---+. (3)|
            ` .   |     ` . |
            (0) ` +---------+ (1)
                     (front)
```

Hence, in the evaluation we will convert our bounding box corners by:

```python
result["bbox3D"] = [
    corners[6],
    corners[4],
    corners[0],
    corners[2],
    corners[7],
    corners[5],
    corners[1],
    corners[3],
]
```
