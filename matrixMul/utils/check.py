#
# FILE      : check.py
#
# BRIEF     : This script parses the .perf files produced by running the
#             script profile.sh and checks whether the experiment passed or failed
#
# AUTHOR    : Waqar Ali (https://github.com/Skidro)

import os, sys, re

def parse (fileName):
    # Create a regex for getting the desired string
    passRegex = "^Result = ([A-Z]+)$"
    # Open and read the file
    with open (fileName, 'r') as fdi:
        for line in fdi:
            passMatch = re.match (passRegex, line)

            # Check if we are on the desired line
            if passMatch:
                # Get the result string
                result = passMatch.group (1)

                # Print the string and exit
                print result
                sys.exit ()

    # Data could not be extraced. Inform the caller
    print "Unexpected data in file : %s" % (fileName)

    # Return to caller
    return
                
def main (fileName):
    # Verify that the file exists
    if not os.path.isfile (fileName):
        print "File (%s) does not exist" % (fileName)
        sys.exit ()

    # Parse the file
    parse (fileName)

    return

# Invoke main when this file is called
if __name__ == "__main__":
    if len (sys.argv) == 2:
        main (sys.argv[1])
    else:
        print "Usage: python check.py <file-name>"
        sys.exit ()
