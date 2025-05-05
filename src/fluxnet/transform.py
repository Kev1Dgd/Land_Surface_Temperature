def reshape_fluxnet_availability(df):
    """Transforme le tableau large de disponibilité Fluxnet en format long exploitable."""
    melted = df.melt(id_vars=["Year/Site ID"], var_name="year", value_name="available")
    melted["available"] = melted["available"] == "+"
    melted = melted.rename(columns={"Year/Site ID": "site_id"})
    melted["year"] = melted["year"].astype(int)
    return melted
