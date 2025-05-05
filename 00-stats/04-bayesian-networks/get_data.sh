if [ -f SMSSpamCollection ]; then
    echo "File already exists"
else
    if [ -f sms+spam+collection.zip ]; then
        echo "Unzip file..."
    else
        echo "Download..."
        wget https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
        echo "Unzip file..."
    fi

    unzip sms+spam+collection.zip
    rm sms+spam+collection.zip readme
fi