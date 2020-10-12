def get_loc_test(get_loc):
    test_points = [(120, 520), (1440, 430), (3210, 740), (410, 1660), (1910, 1850), (3900,1590), (70, 3150), (1720, 3200), (3120, 3650)]

    correct = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]

    parsed = [get_loc(x[0], x[1]) for x in test_points]

    print("\nTesting function get_loc...\n")

    corr = 1

    for i in range(0, 9):
        print("Input:", test_points[i], "Correct answer:", correct[i], "Your answer:", parsed[i])
        if (correct[i] != parsed[i]):
            corr = 0

    if (corr):
        print("\nAll get_loc tests passed!\n")
    else:
        print("\nTest cases failed! Go back and check your get_loc function.\n")
