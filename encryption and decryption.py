from cryptography.fernet import Fernet
import pandas as pd
import pprint
import io

# Generate a Fernet key
key = Fernet.generate_key()
# Create a Fernet instance with the key
fernet = Fernet(key)
# Read the ml-latest-small dataset into a Pandas dataframe
df = pd.read_csv('ml-latest-small/ratings.csv')
# Display the first 5 unencrypted lines
print("Unencrypted lines:")
print(df.head())
# Convert the dataframe to a string and encode it as bytes
data = df.to_csv(index=False).encode()
# Encrypt the data using Fernet
encrypted_data = fernet.encrypt(data)
# Display the first 5 encrypted lines
print("Encrypted lines:")
print(encrypted_data[:len(df.head().to_csv(index=False).encode())].decode())
# Decrypt the data using Fernet
decrypted_data = fernet.decrypt(encrypted_data)
# Convert the decrypted data back into a Pandas dataframe
df_decrypted = pd.read_csv(io.BytesIO(decrypted_data))
# Display the decrypted data using pprint
print("Decrypted lines:")
pprint.pprint(df_decrypted.head().to_dict())

