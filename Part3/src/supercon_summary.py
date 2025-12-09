import h2o
from h2o import H2OFrame

def summarize_frame(frame, name):
    print("\n" + "="*80)
    print(f"SUMMARY FOR: {name}")
    print("="*80)

    # Basic info
    print(f"\nShape (rows, cols): {frame.nrows}, {frame.ncols}")

    # Column names
    print("\nColumn Names:")
    print(frame.col_names)

    # Data types
    print("\nColumn Types:")
    print(frame.types)

    # Missing values
    print("\nMissing Values:")
    try:
        print(frame.isna().sum())
    except:
        print("Cannot compute missing values (older H2O version).")

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(frame.describe())

    print("\n" + "="*80 + "\n")


def main():

    h2o.init()

    # ---- Load the datasets ----
    try:
        train = h2o.import_file("train.csv")
        summarize_frame(train, "TRAIN.CSV")
    except Exception as e:
        print("Error loading train.csv:", e)

    try:
        unique = h2o.import_file("unique_m.csv")
        summarize_frame(unique, "UNIQUE_M.CSV")
    except Exception as e:
        print("Error loading unique_m.csv:", e)


if __name__ == "__main__":
    main()