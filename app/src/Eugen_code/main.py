import cv2
import matcher


def main():
    # read images
    img1 = cv2.imread('data/gt/im3.jpg')
    img2 = cv2.imread('data/gt/im4.jpg')

    # read contours
    contours1 = matcher.read_contours('data/gt/cells3_1.csv')
    contours2 = matcher.read_contours('data/gt/cells4_1.csv')

    # scale contours
    for it in contours1:
        for p in it:
            p[0] = int(p[0] * 2.66)
            p[1] = int(p[1] * 2.66)

    for it in contours2:
        for p in it:
            p[0] = int(p[0] * 2.66)
            p[1] = int(p[1] * 2.66)

    # draw contours
    viz1 = img1.copy()
    viz2 = img2.copy()

    for i in range(len(contours1)):
        viz1 = cv2.drawContours(viz1, contours1, i, (0, 0, 255), 2)

    for i in range(len(contours2)):
        viz2 = cv2.drawContours(viz2, contours2, i, (0, 0, 255), 2)

    # convert contours to contours data
    c1 = matcher.get_contours_data(contours1)
    c2 = matcher.get_contours_data(contours2)

    # get triplets from contours data
    triplets1 = matcher.extract_triplets(c1)
    triplets2 = matcher.extract_triplets(c2)

    # match triplets
    matches = list()
    for i in range(len(triplets1)):

        best_score = 1e10
        best_j = -1

        for j in range(i, len(triplets2)):

            score = matcher.match_triplets(triplets1[i], triplets2[j])

            if score < best_score:
                best_score = score
                best_j = j

        matches.append(matcher.Match(i, best_j, best_score))

    # sort matches
    matches.sort(key=lambda x: x.dist, reverse=False)

    # get best 10 best matched triplets
    matches = matches[0:10]

    # draw matches
    match_im = matcher.draw_matched_contours(img1, c1, img2, c2, triplets1, triplets2, matches)
    cv2.imwrite('output/matches.png', match_im)


if __name__ == '__main__':
    main()
