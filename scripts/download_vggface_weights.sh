mkdir './scripts/weights/'
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU" \
        -O 'scripts/weights/resnet50_ft_weight.pkl' && rm -rf /tmp/cookies.txt