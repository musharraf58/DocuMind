# Larkspur Home: Warehouse Operations Runbook

Larkspur Home is a fictional company invented for the DocuMind retrieval
evaluation. Every name, number and rule in this document is made up.

## Purpose

This runbook is for the night shift at the Tilburg warehouse. It lists the
error codes shown by the sorting line, the handheld scanners and the label
printers, and what to do about each one. Escalate anything not listed here to
the on-call shift lead.

## Handheld scanner errors

ERR-4471 means the scanner lost its connection to the stock database in the
middle of a scan. Reseat the battery, wait for the green light, and scan the
item again. If ERR-4471 appears three times within ten minutes on different
scanners, the warehouse Wi-Fi access point in aisle 12 is probably down: power
cycle it from the switch cabinet behind the loading dock.

ERR-1180 means the barcode was read but the item is not in today's pick list.
Do not force the scan. Put the item in the yellow quarantine crate and record
the bin location on the clipboard next to the crate.

ERR-2093 is a low battery warning. Swap the battery at the charging rack; the
scanner keeps the current pick list for two minutes without power.

## Sorting line faults

A red flashing light at a diverter means a parcel is jammed. Press the
emergency stop at the nearest station before reaching into the line, never
just the pause button. Clear the jam, then reset the line from the main panel
with the blue key. The line restarts at a crawl for 60 seconds as a safety
check.

If the line stops with code SL-77 on the main panel, the weight scale at the
induction point has drifted. Recalibrate it with the 5 kilogram test weight
kept in the drawer under the panel, following the steps printed on the inside
of the drawer.

## Label printer problems

Smudged labels usually mean the print head is dirty. Clean it with one of the
isopropyl wipes from the blue box; never use water. If labels come out blank,
the thermal ribbon has run out: spare ribbons are on the top shelf of cabinet
C3.

A label that prints the wrong carrier logo means the carrier rules were not
refreshed after the 18:00 update. Restart the label service from the shift
lead's laptop; it takes about four minutes and parcels already labelled do not
need to be relabelled.

## End of shift

Before leaving, put every scanner in the charging rack, empty the quarantine
crate into the exceptions cage, and write the number of quarantined items in
the shift log. The morning shift cannot release the outbound trucks until the
shift log is signed.

## Safety

Hi-vis vests are required everywhere past the yellow line. Forklifts have
right of way in the aisles. Report every near miss in the safety log, even if
nobody was hurt; near-miss reports are anonymous.
