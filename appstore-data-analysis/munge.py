import json
import re

f = open("android_reviews.dump")
content = f.read()
apps = json.loads(content)

f = open("android_refined", 'wb')

failed = 0
for app in apps:
    id = app['Url'].split('?')[-1][3:]
    if not app['Reviews']:
        continue
    for review in app['Reviews']:
        try:
            f.write(id + " " + review['reviewBody'].lower())
            f.write("\n")
        except:
            print "failed - " + id
            failed += 1

print "failed: " + str(failed)
