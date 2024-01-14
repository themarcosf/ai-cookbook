"""
This file contains the code to scrape linkedin profiles
using the following API provider: https://nubela.co/proxycurl/api/v2/linkedin

During development, an example profile saved in the filesystem will be used
"""
import os
import json


def scrape_linkedin_profile(linkedin_url: str):
    """scrape information from linkedin profile"""
    file_path = f"{os.path.dirname(__file__)}/proxycurl.json"

    try:
        with open(file_path, "r") as file:
            file_contents = json.loads(file.read())

            # clean up the data
            data = {
                k: v
                for k, v in file_contents.items()
                if v not in ["", None, []]
                and k not in ["people_also_viewed", "connections"]
            }

            if data.get("groups"):
                for group_data in data.get("groups"):
                    group_data.pop("profile_pic_url")

            return data

    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
