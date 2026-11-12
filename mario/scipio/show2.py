import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d.items():
    if isinstance(v, dict) and "all" in v:
        print(f"{k:20s} all {v['all']} crit {v['critical']} repeats {v['when_label_repeats']} changes {v['when_label_changes']} copies_prev_when_wrong {v['copies_prev_when_wrong']} changed {v.get('answers_changed')} chg_share {v['label_changes_share']}")
    else:
        print(k, v)
