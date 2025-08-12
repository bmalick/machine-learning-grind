#!/bin/bash

PWD=$(pwd)
SAVE_DIR=$PWD/papers

# mkdir -p $SAVE_DIR

CHESS=(
    https://arxiv.org/pdf/1712.01815
)

FOOTBALL=(
    https://arxiv.org/pdf/1406.2199
    https://arxiv.org/pdf/1703.06870
    https://arxiv.org/pdf/2103.14196
    https://arxiv.org/pdf/1703.07402
    https://arxiv.org/pdf/2006.04071
    https://arxiv.org/pdf/2004.01888
    https://arxiv.org/pdf/2110.06864
)

GENERATIVE_MODELS=(
    https://arxiv.org/pdf/1406.2661
    https://arxiv.org/pdf/1411.1784
    https://arxiv.org/pdf/1511.06434
    https://arxiv.org/pdf/1606.03657
    https://arxiv.org/pdf/1609.04802
    https://arxiv.org/pdf/1611.07004
    https://arxiv.org/pdf/1701.07875
    https://arxiv.org/pdf/1710.10196
    https://arxiv.org/pdf/1703.10593
    https://arxiv.org/pdf/1711.09020
    https://arxiv.org/pdf/1704.00028
    https://arxiv.org/pdf/1802.05957
    https://arxiv.org/pdf/1812.04948
    https://arxiv.org/pdf/1809.11096
    https://arxiv.org/pdf/1803.07422
    https://arxiv.org/pdf/1912.04958
    https://arxiv.org/pdf/2106.12423
    https://arxiv.org/pdf/1505.05770
    https://arxiv.org/pdf/1906.02735
    https://arxiv.org/pdf/1807.03039
    https://arxiv.org/pdf/1312.6114
    https://arxiv.org/pdf/1502.04623
    https://arxiv.org/pdf/1611.05013
    https://arxiv.org/pdf/1711.00937
    https://arxiv.org/pdf/1906.00446
    https://arxiv.org/pdf/2106.05931
    https://arxiv.org/pdf/1503.03585
    https://arxiv.org/pdf/2006.11239
    https://arxiv.org/pdf/2102.09672
    https://arxiv.org/pdf/2207.04316
    https://arxiv.org/pdf/2112.10752
    https://arxiv.org/pdf/2205.11487
    https://arxiv.org/pdf/2307.01952
    https://arxiv.org/pdf/2212.09748
)

GRAPHNN=(
    https://arxiv.org/pdf/1312.6203
    https://arxiv.org/pdf/1403.6652
    https://arxiv.org/pdf/1607.00653
    https://arxiv.org/pdf/1606.09375
    https://arxiv.org/pdf/1609.02907
    https://arxiv.org/pdf/1706.02216
    https://arxiv.org/pdf/1710.10903
    https://arxiv.org/pdf/1703.06103
    https://arxiv.org/pdf/1806.01973
    https://arxiv.org/pdf/1801.07455
    https://arxiv.org/pdf/1809.10341
    https://arxiv.org/pdf/1810.05997
    https://arxiv.org/pdf/1810.00826
    https://arxiv.org/pdf/1806.03536
    https://arxiv.org/pdf/1907.04931
    https://arxiv.org/pdf/1911.06455
    https://arxiv.org/pdf/1905.07953
    https://arxiv.org/pdf/2007.02133
    https://arxiv.org/pdf/2009.05923
    https://arxiv.org/pdf/2006.04131
    https://arxiv.org/pdf/2006.05582
    https://arxiv.org/pdf/2106.05234
    https://arxiv.org/pdf/2106.05234
    https://arxiv.org/pdf/2205.12454
    https://arxiv.org/pdf/2303.12712
)

IMAGE_CAP=(
    https://arxiv.org/pdf/1904.00373
    https://arxiv.org/pdf/1812.02353
    https://arxiv.org/pdf/1912.08226
    https://arxiv.org/pdf/2003.08129
    https://arxiv.org/pdf/2004.06165
    https://arxiv.org/pdf/1909.11740
    https://arxiv.org/pdf/2101.00529
    https://arxiv.org/pdf/2101.00529
    https://arxiv.org/pdf/2111.09734
    https://arxiv.org/pdf/2201.12086
    https://arxiv.org/pdf/2204.14198
    https://arxiv.org/pdf/2209.06794
    https://arxiv.org/pdf/2301.12597
    https://arxiv.org/pdf/1412.4729
    https://arxiv.org/pdf/1412.6632
    https://arxiv.org/pdf/1412.6448
    https://arxiv.org/pdf/1411.4555
    https://arxiv.org/pdf/1511.07571
    https://arxiv.org/pdf/1502.03044
    https://arxiv.org/pdf/1612.01887
    https://arxiv.org/pdf/1612.00563
    https://arxiv.org/pdf/1707.07998
)

NLP=(
    https://aclanthology.org/A00-1031.pdf
    https://arxiv.org/pdf/1301.3781
    https://aclanthology.org/D14-1162.pdf
    https://arxiv.org/pdf/1406.1078
    https://arxiv.org/pdf/1508.06615
    https://arxiv.org/pdf/1409.0473
    https://arxiv.org/pdf/1706.03762
    https://arxiv.org/pdf/1708.02182
    https://arxiv.org/pdf/1810.04805
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
    https://arxiv.org/pdf/1906.08237
    https://arxiv.org/pdf/2005.14165
    https://arxiv.org/pdf/2001.08361
    https://arxiv.org/pdf/2005.11401
    https://arxiv.org/pdf/2002.08909
    https://arxiv.org/pdf/2005.14165
    https://arxiv.org/pdf/1910.10683
    https://arxiv.org/pdf/2203.02155
    https://arxiv.org/pdf/2201.11903
    https://arxiv.org/pdf/2203.02155
    https://arxiv.org/pdf/2201.11903
    https://arxiv.org/pdf/2206.07682
    https://arxiv.org/pdf/2210.03629
    https://arxiv.org/pdf/2212.08073
    https://arxiv.org/pdf/2203.02155
    https://arxiv.org/pdf/2309.17421
    https://arxiv.org/pdf/2401.04088
    https://arxiv.org/pdf/2310.06825
    https://arxiv.org/pdf/2302.13971
    https://arxiv.org/pdf/2302.04761
    https://arxiv.org/pdf/2404.08567
)

OBJECT_TRACKING=(
    https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    https://arxiv.org/pdf/1312.6229
    https://arxiv.org/pdf/1311.2524
    https://arxiv.org/pdf/1504.08083
    https://arxiv.org/pdf/1506.01497
    https://arxiv.org/pdf/1506.02640
    https://arxiv.org/pdf/1512.02325
    https://arxiv.org/pdf/1708.02002
    https://arxiv.org/pdf/1703.07402
    https://arxiv.org/pdf/2005.12872
    https://arxiv.org/pdf/2012.15460
    https://arxiv.org/pdf/2004.01888
    https://arxiv.org/pdf/2101.02702
    https://arxiv.org/pdf/2110.06864
    https://arxiv.org/pdf/2105.03247
    https://arxiv.org/pdf/2203.14360
)

SEGMENTATION=(
    https://www.csd.uwo.ca/~yboykov/Papers/iccv01.pdf
    https://arxiv.org/pdf/1411.4038
    https://arxiv.org/pdf/1511.00561
    https://arxiv.org/pdf/1412.7062
    https://arxiv.org/pdf/1505.04597
    https://arxiv.org/pdf/1606.06650
    https://arxiv.org/pdf/1606.04797
    https://arxiv.org/pdf/1606.00915
    https://arxiv.org/pdf/1612.01105
    https://arxiv.org/pdf/1611.06612
    https://arxiv.org/pdf/1706.05587
    https://arxiv.org/pdf/1703.06870
    https://arxiv.org/pdf/1802.02611
    https://arxiv.org/pdf/1807.10165
    https://arxiv.org/pdf/1905.04804
    https://arxiv.org/pdf/1904.02689
    https://arxiv.org/pdf/1901.02446
    https://arxiv.org/pdf/2005.12872
    https://arxiv.org/pdf/2001.00309
    https://arxiv.org/pdf/2107.06278
    https://arxiv.org/pdf/2112.01527
    https://arxiv.org/pdf/2105.05633
    https://arxiv.org/pdf/1809.10486
    https://arxiv.org/pdf/2304.02643
    https://arxiv.org/pdf/2304.06718
    https://arxiv.org/pdf/2211.06220
)

SPEECH=(
    https://www.cs.toronto.edu/~graves/icml_2006.pdf
    https://arxiv.org/pdf/1508.01211
    https://arxiv.org/pdf/1609.03499
    https://arxiv.org/pdf/1703.10135
    https://arxiv.org/pdf/1804.03619
    https://arxiv.org/pdf/1712.05884
    https://arxiv.org/pdf/1905.09263
    https://arxiv.org/pdf/1904.03288
    https://arxiv.org/pdf/2005.08100
    https://arxiv.org/pdf/2006.11477
    https://arxiv.org/pdf/2006.04558
    https://arxiv.org/pdf/2005.11129
    https://arxiv.org/pdf/2106.06103
    https://arxiv.org/pdf/2106.07447
    https://arxiv.org/pdf/2212.04356
    https://arxiv.org/pdf/2112.02418
    https://arxiv.org/pdf/2306.15687
)

download(){

    local name=$1
    shift
    local urls=$@
    local name_save_dir=$SAVE_DIR/$name
    mkdir -p $name_save_dir

    echo
    echo Downloading $name subject papers to $name_save_dir
    echo

    i=1
    for url in ${urls[0]}; do
        fname="$name_save_dir/$(printf '%02d' $i).pdf"
        if [ $fname ]; then
            echo "$fname already exists"
        else
            wget -O $fname $url
        fi
        ((i++))
    done
    echo $i papers saved into $name_save_dir

}

download chess ${CHESS[@]}
download football ${FOOTBALL[@]}
download image-generation ${GENERATIVE_MODELS[@]}
download graphs ${GRAPHNN[@]}
download image-captioning ${IMAGE_CAP[@]}
download nlp ${NLP[@]}
download object-tracking ${OBJECT_TRACKING[@]}
download segmentation ${SEGMENTATION[@]}
download speech ${SPEECH[@]}







