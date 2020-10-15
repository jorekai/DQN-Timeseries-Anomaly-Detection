def pretty_print_df(dataframe, head=True):
    if head:
        return dataframe.head(5).to_markdown()
    return dataframe.to_markdown()
