def get_neighbours(idx, idy):
    idx_plus = (idx + 1 if idx != 24 else 0)
    idx_minus = (idx - 1 if idx != 0 else 24)
    idy_plus = (idy + 1 if idy != 24 else 0)
    idy_minus = (idy - 1 if idy != 0 else 24)

    return [[idx, idy_minus], [idx, idy_plus], [idx_minus, idy], [idx_plus, idy], [idx_minus, idy_minus],
            [idx_minus, idy_plus], [idx_plus, idy_minus], [idx_plus, idy_plus]]


def check_rules(start, alive_cell,  neighbours_indices):
    live_neighbours = sum([start[x[0], x[1]] for x in neighbours_indices])

    if alive_cell:
        # Cell is alive
        if live_neighbours < 2:
            return 0
        elif live_neighbours > 3:
            return 0
        else:
            return 1

    else:
        # Cell is dead
        if live_neighbours == 3:
            return 1
        else:
            return 0


def conway(start):
    end = start.copy()
    for idx, row in enumerate(start):
        for idy, column in enumerate(row):
            neighbours_indices = get_neighbours(idx, idy)
            new_value = check_rules(start, alive_cell=column, neighbours_indices=neighbours_indices)
            end[idx, idy] = new_value

    return end


def conway_steps(start, delta):
    end = start.copy()
    for i in range(delta):
        end = conway(end)
    return end
