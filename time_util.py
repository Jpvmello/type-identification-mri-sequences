def time_format(total_seconds):
    hours = int(total_seconds/3600)
    minutes = int((total_seconds/3600 - hours)*60)
    seconds = int(total_seconds - 3600*hours - 60*minutes)
    return str(hours) + 'h' + str(minutes) + 'min' + str(seconds) + 's'
