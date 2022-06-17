def rows_to_cols(rows, edge_len, pad_width):
    """Utility for padding"""
    top_edge = []
    for i in range(edge_len):
        col = []
        for j in range(pad_width):
            col.append(rows[j][i])
        top_edge.append(col)
    return top_edge


def cols_to_rows(cols, pad_width):
    """Utility for padding"""
    edge = []
    for i in range(pad_width):
        row = [x[i] for x in cols]
        edge.append(row)
    return edge


def pad_column(column, pad_width):
    """Utility for padding"""
    padded = []
    for col in column:
        col_pad = pad_width * [col[0]] + col + pad_width * [col[-1]]
        padded.append(col_pad)
    return padded


def create_edge_dict(magnitudes, pad_width):
    """Utility for padding"""
    panel_edges = []
    for panel in range(5):
        left = magnitudes[panel][:pad_width]  # get left column(s) (all lowest x)
        right = magnitudes[panel][-pad_width:]  # get right column(s) (all highest x)
        bottom = [x[:pad_width] for x in magnitudes[panel]]  # get bottom row(s) (all lowest y)
        top = [x[-pad_width:] for x in magnitudes[panel]]  # get top row(s) (all highest y)
        edge_dict = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        panel_edges.append(edge_dict)
    return panel_edges


def pad_equatorial_panel(magnitudes_panel, panel, panel_edges, edge_len, pad_width):
    """add padding to equatorial panels"""
    # get row/column from panel 4 to pad top edge
    if panel == 0:
        # no need to reverse at all
        top_edge = panel_edges[4]['bottom'].copy()
    elif panel == 1:
        # need to reverse in only one direction
        top_edge = rows_to_cols(panel_edges[4]['right'].copy(), edge_len=edge_len, pad_width=pad_width)
        top_edge = [list(reversed(col)) for col in top_edge]
    elif panel == 2:
        # need to reverse in both directions
        top_edge = panel_edges[4]['top'].copy()
        top_edge.reverse()
        top_edge = [list(reversed(col)) for col in top_edge]
    else:
        # need to reverse in only one direction
        top_edge = rows_to_cols(panel_edges[4]['left'].copy(), edge_len=edge_len, pad_width=pad_width)
        top_edge.reverse()

    # pad TOP AND BOTTOM of panel on a column by column basis
    # pad bottom of column via replication
    column_list = []
    for i in range(edge_len):
        column = magnitudes_panel[i]
        col_pad = pad_width * [column[0]] + column + top_edge[i]
        column_list.append(col_pad)

    # pad LEFT AND RIGHT side of each panel around horizontal plane
    # get panel index for left and right panel
    left_panel = (panel - 1) % 4
    right_panel = (panel + 1) % 4

    # get the rightmost column of the left panel, and pad top and bottom with edge values
    left_col = panel_edges[left_panel]['right']
    left_col_pad = pad_column(left_col, pad_width)

    # get the leftmost column of the right panel, and pad top and bottom with edge values
    right_col = panel_edges[right_panel]['left']
    right_col_pad = pad_column(right_col, pad_width)

    # COMBINE left column, padded center columns, and right column to get final version
    return left_col_pad + column_list + right_col_pad


def pad_top_panel(magnitudes_top, panel_edges, edge_len, pad_width):
    """add padding to top panel"""
    # pad TOP AND BOTTOM of panel on a column by column basis
    column_list = []
    bottom_edge = panel_edges[0]['top'].copy()
    top_edge = panel_edges[2]['top'].copy()
    top_edge.reverse()
    top_edge = [list(reversed(col)) for col in top_edge]
    for i in range(edge_len):
        column = magnitudes_top[i]
        col_pad = bottom_edge[i] + column + top_edge[i]
        column_list.append(col_pad)

    # get the top row of panel 3, reverse it, and pad top and bottom with edge values
    left_col = panel_edges[3]['top'].copy()
    left_col.reverse()
    left_col_pad = pad_width * [left_col[0]] + left_col + pad_width * [left_col[-1]]

    # get the top row of panel 1, and pad top and bottom with edge values
    right_col = panel_edges[1]['top'].copy()
    right_col_pad = pad_width * [right_col[0]] + right_col + pad_width * [right_col[-1]]
    right_col_pad = [list(reversed(col)) for col in right_col_pad]

    # convert from columns to rows
    left_col_pad = cols_to_rows(left_col_pad, pad_width)
    right_col_pad = cols_to_rows(right_col_pad, pad_width)

    # COMBINE left column, padded center columns, and right column to get final version
    return left_col_pad + column_list + right_col_pad


def pad_cubed_sphere(magnitudes: list, pad_width: int):
    """Add padding to each of the 5 faces of the cubed sphere, based on values from the adjacent face

    :param magnitudes: A list of panels, where each element of the list represents one panel of the cube sphere
    :param pad_width: How much padding to add on each side of each panel
    """
    edge_len = len(magnitudes[0])

    # create a list of dictionaries (one for each panel) containing the left, right, top and bottom edges for each panel
    panel_edges = create_edge_dict(magnitudes, pad_width)

    # create empty list of lists of lists
    magnitudes_pad = [[[[] for _ in range(edge_len + 2 * pad_width)] for _ in range(edge_len + 2 * pad_width)]
                      for _ in range(5)]

    # diagram of unfolded cube, with panel indices
    #             _______
    #            |       |
    #            |   4   |
    #     _______|_______|_______ _______
    #    |       |       |       |       |
    #    |   3   |   0   |   1   |   2   |
    #    |_______|_______|_______|_______|
    # In all cases, low values of x and y are situated in lower left of the unfolded sphere

    # pad the 4 panels around the horizontal plane
    for panel in range(4):
        magnitudes_pad[panel] = pad_equatorial_panel(magnitudes[panel], panel, panel_edges, edge_len, pad_width)

    # now pad for the top panel (panel 4)
    magnitudes_pad[4] = pad_top_panel(magnitudes[4], panel_edges, edge_len, pad_width)

    return magnitudes_pad
