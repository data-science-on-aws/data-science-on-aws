aws s3 sync \
    . \
    s3://dsoaws/workshop/ \
    --follow-symlinks \
    --delete \
    --acl public-read-write \
    --acl bucket-owner-full-control \
    --exclude "*" \
    --include "00_quickstart/*" 

