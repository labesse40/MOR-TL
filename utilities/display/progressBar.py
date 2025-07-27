import sys

def progressBar(count_value=None, total=None, bar_length=10, suffix=''):
    """
    Displays a progress bar in the console.
    This function generates a textual progress bar that updates dynamically
    as the progress increases. 
    Args:
        count_value (int or float, optional): The current progress value. Must be less than or equal to `total`.
        total (int or float, optional): The total value representing 100% progress.
        bar_length (int, optional): The length of the progress bar in characters. Default is 10.
        suffix (str, optional): A string to display at the end of the progress bar. Default is an empty string.
    Notes:
        - If `count_value` or `total` is not provided, or if they are invalid, the function does nothing.
        - The progress bar is displayed in the format: [====------] 40.0% ...suffix
    Example:
        >>> progressBar(count_value=4, total=10, bar_length=20, suffix='Processing')
        [========------------] 40.0% ...Processing
    """

    if (count_value is not None and total is not None) and isinstance(count_value, (int, float)) \
            and isinstance(total, (int, float)) and (count_value <= total):
    
        if not isinstance(bar_length, int) or bar_length <= 0:
            bar_length = 10
        if not isinstance(suffix, str):
            suffix = ''

        filled_up_Length = int(round(bar_length* count_value / float(total)))
        percentage = round(100.0 * count_value/float(total),1)
        bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
        sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
        sys.stdout.flush()
    else:
        pass